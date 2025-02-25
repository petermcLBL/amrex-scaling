#include <AMReX.H>
#include <AMReX_FFT.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

namespace {
    constexpr int ntests = 10;
}

template <typename F>
double test_amrex (F& r2c, MultiFab& mf, cMultiFab& cmf)
{
    Gpu::synchronize();
    double t00 = amrex::second();

    r2c.forward(mf, cmf);
    r2c.backward(cmf, mf);

    Gpu::synchronize();
    double t0 = amrex::second();

    amrex::Print() << "    Warm-up: " << t0-t00 << "\n";

    double tt = 0;

    for (int itest = 0; itest < ntests; ++itest) {
        double ta = amrex::second();
        r2c.forward(mf, cmf);
        r2c.backward(cmf, mf);
        Gpu::synchronize();
        double tb = amrex::second();
        tt += (tb-ta);
        amrex::Print() << "    Test # " << itest << ": " << tb-ta << "\n";
    }

    return tt / double(ntests);
}

double test_amrex_pencil (Box const& domain, MultiFab& mf, cMultiFab& cmf)
{
    FFT::Info info{};
    info.setTwoDMode(true);
    info.setDomainStrategy(FFT::DomainStrategy::pencil);
    FFT::R2C<Real,FFT::Direction::both> r2c(domain, info);
    return test_amrex(r2c, mf, cmf);
}

double test_amrex_slab (Box const& domain, MultiFab& mf, cMultiFab& cmf)
{
    FFT::Info info{};
    info.setTwoDMode(true);
    info.setDomainStrategy(FFT::DomainStrategy::slab);
    FFT::R2C<Real,FFT::Direction::both> r2c(domain, info);
    return test_amrex(r2c, mf, cmf);
}

int main (int argc, char* argv[])
{
    static_assert(AMREX_SPACEDIM == 3);

    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main");

        AMREX_D_TERM(int n_cell_x = 256;,
                     int n_cell_y = 256;,
                     int n_cell_z = 256);

        {
            ParmParse pp;
            AMREX_D_TERM(pp.query("n_cell_x", n_cell_x);,
                         pp.query("n_cell_y", n_cell_y);,
                         pp.query("n_cell_z", n_cell_z));
        }

        amrex::Print() << "\n FFT size: " << n_cell_x << " " << n_cell_y << " " << n_cell_z << " "
                       << "  # of proc. " << ParallelDescriptor::NProcs() << "\n\n";

        Box domain(IntVect(0),IntVect(n_cell_x-1,n_cell_y-1,n_cell_z-1));
        BoxArray ba = amrex::decompose(domain, ParallelDescriptor::NProcs(), {true,true,true});

        AMREX_ALWAYS_ASSERT(ba.size() == ParallelDescriptor::NProcs());
        DistributionMapping dm = FFT::detail::make_iota_distromap(ba.size());

        GpuArray<Real,3> dx{1._rt/Real(n_cell_x), 1._rt/Real(n_cell_y), 1._rt/Real(n_cell_z)};

        MultiFab mf(ba, dm, 1, 0);
        auto const& ma = mf.arrays();
        ParallelFor(mf, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            AMREX_D_TERM(Real x = (i+0.5_rt) * dx[0] - 0.5_rt;,
                         Real y = (j+0.5_rt) * dx[1] - 0.5_rt;,
                         Real z = (k+0.5_rt) * dx[2] - 0.5_rt);
            ma[b](i,j,k) = std::exp(-10._rt*
                (AMREX_D_TERM(x*x*1.05_rt, + y*y*0.90_rt, + z*z)));
        });
        Gpu::streamSynchronize();

        Box cdomain(IntVect(0), IntVect(n_cell_x/2, n_cell_y-1, n_cell_z-1));
        BoxArray cba = amrex::decompose(cdomain, ParallelDescriptor::NProcs(), {true,true,true});
        AMREX_ALWAYS_ASSERT(cba.size() == ParallelDescriptor::NProcs());

        cMultiFab cmf(cba, dm, 1, 0);

        auto t_amrex_pencil = test_amrex_pencil(domain, mf, cmf);
        auto t_amrex_slab = test_amrex_slab(domain, mf, cmf);
        amrex::Print() << "  armex pencil time: " << t_amrex_pencil << "\n"
                       << "  amrex slab   time: " << t_amrex_slab << "\n\n";
    }
    amrex::Finalize();
}
