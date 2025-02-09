#include <AMReX.H>
#include <AMReX_FFT.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#if defined(USE_HEFFTE)
#include <heffte.h>
#endif

#if defined(USE_FFTX)
#include <fftx_mpi.hpp>
#endif

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

double test_amrex_auto (Box const& domain, MultiFab& mf, cMultiFab& cmf)
{
    FFT::R2C<Real,FFT::Direction::both> r2c(domain);
    return test_amrex(r2c, mf, cmf);
}

double test_amrex_pencil (Box const& domain, MultiFab& mf, cMultiFab& cmf)
{
    FFT::R2C<Real,FFT::Direction::both> r2c
        (domain, FFT::Info{}.setDomainStrategy(FFT::DomainStrategy::pencil));
    return test_amrex(r2c, mf, cmf);
}

double test_amrex_slab (Box const& domain, MultiFab& mf, cMultiFab& cmf)
{
    FFT::R2C<Real,FFT::Direction::both> r2c
        (domain, FFT::Info{}.setDomainStrategy(FFT::DomainStrategy::slab));
    return test_amrex(r2c, mf, cmf);
}

#ifdef USE_HEFFTE
double test_heffte (Box const& /*domain*/, MultiFab& mf, cMultiFab& cmf)
{
    auto& fab = mf[ParallelDescriptor::MyProc()];
    auto& cfab = cmf[ParallelDescriptor::MyProc()];

    auto const& local_box = fab.box();
    auto const& c_local_box = cfab.box();

#ifdef AMREX_USE_CUDA
    heffte::fft3d_r2c<heffte::backend::cufft> fft
#elif AMREX_USE_HIP
    heffte::fft3d_r2c<heffte::backend::rocfft> fft
#else
    heffte::fft3d_r2c<heffte::backend::fftw> fft
#endif
        ({{local_box.smallEnd(0),local_box.smallEnd(1),local_box.smallEnd(2)},
          {local_box.bigEnd(0)  ,local_box.bigEnd(1)  ,local_box.bigEnd(2)}},
         {{c_local_box.smallEnd(0),c_local_box.smallEnd(1),c_local_box.smallEnd(2)},
          {c_local_box.bigEnd(0)  ,c_local_box.bigEnd(1)  ,c_local_box.bigEnd(2)}},
         0, ParallelDescriptor::Communicator());

    using heffte_complex = typename heffte::fft_output<Real>::type;

    fft.forward(fab.dataPtr(), (heffte_complex*)cfab.dataPtr());
    fft.backward((heffte_complex*)cfab.dataPtr(), fab.dataPtr());

    Gpu::synchronize();
    double t0 = amrex::second();

    for (int itest = 0; itest < ntests; ++itest) {
        fft.forward(fab.dataPtr(), (heffte_complex*)cfab.dataPtr());
        fft.backward((heffte_complex*)cfab.dataPtr(), fab.dataPtr());
    }

    Gpu::synchronize();
    double t1 = amrex::second();

    return (t1-t0) / double(ntests);
}
#endif

#ifdef USE_FFTX
double test_fftx (Box const& domain, MultiFab& mf, cMultiFab& cmf)
{
    auto& fab = mf[ParallelDescriptor::MyProc()];
    auto& cfab = cmf[ParallelDescriptor::MyProc()];

    int batch = 1;
    bool is_embedded = false;
    bool is_complex = false;
    fftx_plan plan = fftx_plan_distributed_1d(ParallelDescriptor::NProcs(),
                                              domain.length(0),
                                              domain.length(1),
                                              domain.length(2),
                                              batch, is_embedded, is_complex);

    // The code here is likely wrong, because ...

    // The arrays have the Fortran column-major order. For both mf and cmf, the order is
    // (x,y,z) and the domain decomposition is in the z-direction.

    // The comments in fftx/examples/3DDFT_mpi/test3DDFT_mpi_1D.cpp seem to suggest that
    // the output complex array has the order of (z,x,y) and the domain decompostion is in
    // the x-direction.

    // How do we fix it?

    Gpu::synchronize();
    double t00 = amrex::second();

    fftx_execute_1d(plan, (double*)cfab.dataPtr(), fab.dataPtr(), DEVICE_FFT_FORWARD);
    fftx_execute_1d(plan, fab.dataPtr(), (double*)cfab.dataPtr(), DEVICE_FFT_INVERSE);

    Gpu::synchronize();
    double t0 = amrex::second();

    amrex::Print() << "    Warm-up: " << t0-t00 << "\n";

    for (int itest = 0; itest < ntests; ++itest) {
        double ta = amrex::second();
        fftx_execute_1d(plan, (double*)cfab.dataPtr(), fab.dataPtr(), DEVICE_FFT_FORWARD);
        fftx_execute_1d(plan, fab.dataPtr(), (double*)cfab.dataPtr(), DEVICE_FFT_INVERSE);
        Gpu::synchronize();
        double tb = amrex::second();
        amrex::Print() << "    Test # " << itest << ": " << tb-ta << "\n";
    }

    Gpu::synchronize();
    double t1 = amrex::second();

    fftx_plan_destroy(plan);

    return (t1-t0) / double(ntests);
}
#endif

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
#if defined(USE_FFTX)
        // FFTX support 1d or 2d grid. 1D grid distribution assumes that all p processors
        // are logically organized into a linear array. The entire 3D FFT is distributed
        // along the Z dimension of the FFT. Using this processor grid, the X dimension of
        // the FFT is assumed to be laid out consecutively in local memory.
        BoxArray ba = amrex::decompose(domain, ParallelDescriptor::NProcs(), {false,false,true});
#else
        BoxArray ba = amrex::decompose(domain, ParallelDescriptor::NProcs(), {true,true,true});
#endif
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
#if defined(USE_FFTX)
        BoxArray cba = amrex::decompose(cdomain, ParallelDescriptor::NProcs(), {false,false,true});
#else
        BoxArray cba = amrex::decompose(cdomain, ParallelDescriptor::NProcs(), {true,true,true});
#endif
        AMREX_ALWAYS_ASSERT(cba.size() == ParallelDescriptor::NProcs());

        cMultiFab cmf(cba, dm, 1, 0);

        auto t_amrex_auto = test_amrex_auto(domain, mf, cmf);
        auto t_amrex_pencil = test_amrex_pencil(domain, mf, cmf);
        auto t_amrex_slab = test_amrex_slab(domain, mf, cmf);
        amrex::Print() << "  armex atuo   time: " << t_amrex_auto << "\n"
                       << "  armex pencil time: " << t_amrex_pencil << "\n"
                       << "  amrex slab   time: " << t_amrex_slab << "\n";

#ifdef USE_HEFFTE
        auto t_heffte = test_heffte(domain, mf, cmf);
        amrex::Print() << "  heffte       time: " << t_heffte << "\n";
#endif

#ifdef USE_FFTX
        auto t_fftx = test_fftx(domain, mf, cmf);
        amrex::Print() << "  fftx         time: " << t_fftx << "\n";
#endif

        amrex::Print() << "\n";
    }
    amrex::Finalize();
}
