#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv); {

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

        Box domain(IntVect(0),IntVect(n_cell));
        RealBox rb({0.,0.,0.},{1.,1.,1.});
        Array<int,3> is_periodic{1,1,1};
        geom.define(domain, rb, CoordSys::cartesian, is_periodic);

        ba.define(domain);
        ba.maxSize(max_grid_size);

        dm.define(ba);
    }

    MultiFab E;
    MultiFab B;
    MultiFab current;
    MultiFab rho;
    

    } amrex::Finalize();
}
