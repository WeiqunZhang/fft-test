#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_LayoutData.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuComplex.H>

#include <heffte.h>

using namespace amrex;
//using namespace HEFFTE;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv); {

    BL_PROFILE("main");

    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
    IntVect nghost;
    {
        ParmParse pp;
        IntVect n_cell;
        IntVect max_grid_size;
        pp.get("n_cell", n_cell);
        pp.get("max_grid_size", max_grid_size);
        pp.get("nghost", nghost);

        Box domain(IntVect(0),n_cell-IntVect(1));
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
        amrex::ParallelForRNG(mfi.fabbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k, RandomEngine const& engine) noexcept
        {
            fab(i,j,k) = amrex::Random(engine);
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
                    b.growLo(idim,nghost[idim]);
                }
                if (b.bigEnd(idim) == geom.Domain().bigEnd(idim)) {
                    b.growHi(idim,nghost[idim]);
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
    MultiFab real_field(real_ba,dm,1,0,MFInfo().SetArena(The_Device_Arena()));
    real_field.setVal(0.0+ParallelDescriptor::MyProc()); // touch the memory

    Box r_local_box = amrex::shift(my_domain, nghost);
    Box c_local_box = amrex::coarsen(r_local_box, IntVect(2,1,1));
    if (c_local_box.bigEnd(0) * 2 == r_local_box.bigEnd(0)) {
        c_local_box.setBig(0,c_local_box.bigEnd(0)-1);// to avoid overlap
    }
    if (my_domain.bigEnd(0) == geom.Domain().bigEnd(0) + nghost[0]) {
        c_local_box.growHi(0,1);
    }

    BaseFab<GpuComplex<Real> > spectral_field(c_local_box, 1, The_Device_Arena());

#ifdef AMREX_USE_CUDA
    heffte::fft3d_r2c<heffte::backend::cufft> fft
#else
    heffte::fft3d_r2c<heffte::backend::fftw> fft
#endif
        ({{r_local_box.smallEnd(0),r_local_box.smallEnd(1),r_local_box.smallEnd(2)},
          {r_local_box.bigEnd(0)  ,r_local_box.bigEnd(1)  ,r_local_box.bigEnd(2)}},
         {{c_local_box.smallEnd(0),c_local_box.smallEnd(1),c_local_box.smallEnd(2)},
          {c_local_box.bigEnd(0)  ,c_local_box.bigEnd(1)  ,c_local_box.bigEnd(2)}},
         0, ParallelDescriptor::Communicator());

    using heffte_complex = typename heffte::fft_output<Real>::type;
    heffte_complex* spectral_data = (heffte_complex*) spectral_field.dataPtr();

    fft.forward(real_field[my_boxid].dataPtr(), spectral_data);
    fft.backward(spectral_data, real_field[my_boxid].dataPtr());

    ParallelDescriptor::Barrier();

    { BL_PROFILE("HEFFTE-total");
    {
        BL_PROFILE("ForwardTransform");
        fft.forward(real_field[my_boxid].dataPtr(), spectral_data);
    }

    {
        BL_PROFILE("BackwardTransform");
        fft.backward(spectral_data, real_field[my_boxid].dataPtr());
    }
    }

    } amrex::Finalize();
}
