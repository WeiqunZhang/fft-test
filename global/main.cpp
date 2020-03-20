#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_LayoutData.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuComplex.H>

#include <heffte.h>

using namespace amrex;
using namespace HEFFTE;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv); {
    heffte_init();

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

    MultiFab orig_field(ba,dm,1,0,MFInfo().SetArena(The_Device_Arena()));
    for (MFIter mfi(orig_field); mfi.isValid(); ++mfi) {
        Array4<Real> const& fab = orig_field.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            fab(i,j,k) = amrex::Random();
        });
    }

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(orig_field.local_size() == 1, "Must have one Box per process");

    BoxArray real_ba;
    Box my_domain;
    int my_boxid;
    {
        BoxList bl;
        bl.reserve(ba.size());
        for (int i = 0; i < ba.size(); ++i) {
            Box b = ba[i];
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (b.smallEnd(idim) == geom.Domain().smallEnd(idim)) {
                    b.growLo(idim,nghost);
                }
                if (b.bigEnd(idim) == geom.Domain().bigEnd(idim)) {
                    b.growHi(idim,nghost);
                }
            }
            bl.push_back(b);
            if (ParallelDescriptor::MyProc() == dm[i]) {
                my_domain = b;
                my_boxid = i;
            }
        }
        real_ba.define(std::move(bl));
    }
    MultiFab real_field(real_ba,dm,1,0,MFInfo().SetAlloc(false));

    std::unique_ptr<FFT3d<Real> > forward_fft(new FFT3d<Real>(ParallelDescriptor::Communicator()));
    std::unique_ptr<FFT3d<Real> > backward_fft(new FFT3d<Real>(ParallelDescriptor::Communicator()));
    forward_fft->mem_type = HEFFTE_MEM_GPU;
    backward_fft->mem_type = HEFFTE_MEM_GPU;

    Box global_domain = amrex::grow(geom.Domain(), nghost);
    IntVect global_N = global_domain.size();
    IntVect local_lo = my_domain.smallEnd() + IntVect(nghost);
    IntVect local_hi = my_domain.bigEnd() + IntVect(nghost);
    Array<int,3> workspace; // fftsize, sendsize and recvsize

    heffte_plan_r2c_create(forward_fft.get(), global_N.getVect(),
                           local_lo.getVect(), local_hi.getVect(),
                           local_lo.getVect(), local_hi.getVect(),
                           workspace.data());

    Real* dwork_real;
    Real* dwork_spectral;
    int64_t nbytes;
    heffte_allocate(HEFFTE_MEM_GPU, &dwork_real, workspace[0], nbytes);
    heffte_allocate(HEFFTE_MEM_GPU, &dwork_spectral, workspace[0], nbytes);

    real_field.setFab(my_boxid, new FArrayBox(my_domain, 1, dwork_real));
    real_field.setVal(0.0); // touch the memory

    // Warming up runnings
    real_field.ParallelCopy(orig_field, geom.periodicity());
    heffte_execute_r2c(forward_fft.get(), dwork_real, dwork_spectral);
    heffte_execute_r2c(backward_fft.get(), dwork_spectral, dwork_real);

    {
        BL_PROFILE("CopyToRealField");
        real_field.ParallelCopy(orig_field, geom.periodicity());
    }

    {
        BL_PROFILE("ForwardTransform");
        heffte_execute_r2c(forward_fft.get(), dwork_real, dwork_spectral);
    }

    {
        BL_PROFILE("BackwardTransform");
        heffte_execute_r2c(backward_fft.get(), dwork_spectral, dwork_real);
    }

    heffte_deallocate(HEFFTE_MEM_GPU, dwork_real);
    heffte_deallocate(HEFFTE_MEM_GPU, dwork_spectral);

    } amrex::Finalize();
}
