#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_LayoutData.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuComplex.H>

#ifdef AMREX_USE_CUDA
#include <cufft.h>
#else
#include <fftw3.h>
#include <fftw3-mpi.h>
#endif

using namespace amrex;

#ifdef AMREX_USE_CUDA
std::string cufftErrorToString (const cufftResult& err)
{
    switch (err) {
    case CUFFT_SUCCESS:  return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    default: return std::to_string(err) + " (unknown error code)";
    }
}
#endif

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv); {

#ifdef USE_FFTW
    fftw_mpi_init();
#endif

    BL_PROFILE("main");

    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
    int nghost;
    {
        ParmParse pp;
        Vector<int> n_cell;
        int max_grid_size;
        pp.getarr("n_cell", n_cell);
        pp.get("max_grid_size", max_grid_size);
        pp.get("nghost", nghost);

        Box domain(IntVect(0),IntVect(n_cell)-IntVect(1));
        RealBox rb({0.,0.,0.},{1.,1.,1.});
        Array<int,3> is_periodic{1,1,1};
        geom.define(domain, rb, CoordSys::cartesian, is_periodic);

        ba.define(domain);
        ba.maxSize(max_grid_size);

        dm.define(ba);
    }

    MultiFab real_field(ba,dm,1,nghost,MFInfo().SetArena(The_Device_Arena()));
    for (MFIter mfi(real_field); mfi.isValid(); ++mfi) {
        Array4<Real> const& fab = real_field.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            fab(i,j,k) = amrex::Random();
        });
    }

#ifdef AMREX_USE_CUDA
    using FFTplan = cufftHandle;
    using FFTcomplex = cuDoubleComplex;
#else
    using FFTplan = fftw_plan;
    using FFTcomplex = fftw_complex;
#endif
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real> > > > spectral_field;
    Vector<FFTplan> forward_plan;
    Vector<FFTplan> backward_plan;
    for (MFIter mfi(real_field); mfi.isValid(); ++mfi) {
        Box realspace_bx = mfi.fabbox();
        IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT
        IntVect spectral_bx_size = fft_size;
        spectral_bx_size[0] = fft_size[0]/2 + 1;
        Box spectral_bx = Box(IntVect(0), spectral_bx_size - IntVect(1));
        spectral_field.emplace_back(new BaseFab<GpuComplex<Real> >(spectral_bx,1,
                                                                   The_Device_Arena()));
        spectral_field.back()->setVal<RunOn::Device>(0.0); // touch the memory

        FFTplan fplan, bplan;
#ifdef AMREX_USE_CUDA
        cufftResult result = cufftPlan3d(&fplan, fft_size[2], fft_size[1], fft_size[0], CUFFT_D2Z);
        if (result != CUFFT_SUCCESS) {
            amrex::AllPrint() << " cufftplan3d forward failed! Error: "
                              << cufftErrorToString(result) << "\n";
        }

        result = cufftPlan3d(&bplan, fft_size[2], fft_size[1], fft_size[0], CUFFT_Z2D);
        if (result != CUFFT_SUCCESS) {
            amrex::AllPrint() << " cufftplan3d backward failed! Error: "
                              << cufftErrorToString(result) << "\n";
        }
#else
        fplan = fftw_plan_dft_r2c_3d(fft_size[2], fft_size[1], fft_size[0],
                                     real_field[mfi].dataPtr(),
                                     reinterpret_cast<FFTcomplex*>
                                         (spectral_field.back()->dataPtr()),
                                     FFTW_ESTIMATE);

        bplan = fftw_plan_dft_c2r_3d(fft_size[2], fft_size[1], fft_size[0],
                                     reinterpret_cast<FFTcomplex*>
                                         (spectral_field.back()->dataPtr()),
                                     real_field[mfi].dataPtr(),
                                     FFTW_ESTIMATE);
#endif
        forward_plan.push_back(fplan);
        backward_plan.push_back(bplan);
    }

    {
        BL_PROFILE("RealDataFillBoundary");
        real_field.FillBoundary(geom.periodicity());
    }

    // ForwardTransform
    {
        BL_PROFILE("ForwardTransform");
        for (MFIter mfi(real_field); mfi.isValid(); ++mfi)
        {
            int i = mfi.LocalIndex();
#ifdef AMREX_USE_CUDA
            cufftSetStream(forward_plan[i], amrex::Gpu::gpuStream());
            cufftResult result = cufftExecD2Z(forward_plan[i],
                                              real_field[mfi].dataPtr(),
                                              reinterpret_cast<FFTcomplex*>
                                                  (spectral_field[i]->dataPtr()));
            if (result != CUFFT_SUCCESS) {
                amrex::AllPrint() << " forward transform using cufftExec failed! Error: "
                                  << cufftErrorToString(result) << "\n";
            }
#else
            fftw_execute(forward_plan[i]);
#endif
        }
    }

    
    // BackwardTransform
    {
        BL_PROFILE("BackwardTransform");
        for (MFIter mfi(real_field); mfi.isValid(); ++mfi)
        {
            int i = mfi.LocalIndex();
#ifdef AMREX_USE_CUDA
            cufftSetStream(backward_plan[i], amrex::Gpu::gpuStream());
            cufftResult result = cufftExecZ2D(backward_plan[i],
                                              reinterpret_cast<FFTcomplex*>
                                                  (spectral_field[i]->dataPtr()),
                                              real_field[mfi].dataPtr());
            if (result != CUFFT_SUCCESS) {
                amrex::AllPrint() << " backward transform using cufftExec failed! Error: "
                                  << cufftErrorToString(result) << "\n";
            }
#else
            fftw_execute(backward_plan[i]);
#endif
        }
    }

    // destroy fft plans
    for (int i = 0; i < forward_plan.size(); ++i) {
#ifdef AMREX_USE_CUDA
        cufftDestroy(forward_plan[i]);
        cufftDestroy(backward_plan[i]);
#else
        fftw_destroy_plan(forward_plan[i]);
        fftw_destroy_plan(backward_plan[i]);
#endif
    }

#ifdef USE_FFTW
    fftw_mpi_cleanup();
#endif

    } amrex::Finalize();
}
