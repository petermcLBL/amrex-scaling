#include <AMReX.H>
#include <AMReX_FFT.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#if defined(USE_FFTX)
#include <fftx_mpi.hpp>
#endif

using namespace amrex;

namespace {
    constexpr int ntests = 10;
}

template <typename F>
double test_amrex (F& c2c, cMultiFab const& mf, cMultiFab& mf2)
{
    Gpu::synchronize();
    double t00 = amrex::second();

    c2c.forward(mf);
    c2c.backward(mf2);

    Gpu::synchronize();
    double t0 = amrex::second();

    amrex::Print() << "    Warm-up: " << t0-t00 << "\n";

    double tt = 0;

    for (int itest = 0; itest < ntests; ++itest) {
        double ta = amrex::second();
        c2c.forward(mf);
        c2c.backward(mf2);
        Gpu::synchronize();
        double tb = amrex::second();
        tt += (tb-ta);
        amrex::Print() << "    Test # " << itest << ": " << tb-ta << "\n";
    }

    return tt / double(ntests);
}

double test_amrex_auto (Box const& domain, cMultiFab const& mf, cMultiFab& mf2)
{
    FFT::Info info{};
    info.setDomainStrategy(FFT::DomainStrategy::automatic);
    info.setBatchSize(mf.nComp());
    FFT::C2C<Real,FFT::Direction::both> c2c(domain, info);
    return test_amrex(c2c, mf, mf2);
}

#ifdef USE_FFTX
double test_fftx (Box const& domain, cMultiFab const& mf, cMultiFab& mf2)
{
    auto& fab = mf[ParallelDescriptor::MyProc()];
    auto& fab2 = mf2[ParallelDescriptor::MyProc()];

    amrex::Print(Print::AllProcs) << "rank " << ParallelDescriptor::MyProc()
                                  << " fabs on boxes "
                                  << fab.box() << " and " << fab2.box() << std::endl;
      
    int batch = 1;
    bool is_embedded = false;
    bool is_complex = true;
    fftx_plan plan = fftx_plan_distributed_1d(MPI_COMM_WORLD,
                                              ParallelDescriptor::NProcs(),
                                              domain.length(0),
                                              domain.length(1),
                                              domain.length(2),
                                              batch, is_embedded, is_complex);

    // The code here is likely wrong, because ...

    // The arrays have the Fortran column-major order. For both mf and mf2, the order is
    // (x,y,z) and the domain decomposition is in the z-direction.

    // The comments in fftx/examples/3DDFT_mpi/test3DDFT_mpi_1D.cpp seem to suggest that
    // the output complex array has the order of (z,x,y) and the domain decompostion is in
    // the x-direction.

    // How do we fix it?

    Gpu::synchronize();
    double t00 = amrex::second();

    // plan, output, input, direction
    fftx_execute_1d(plan, (double*) fab2.dataPtr(), (double*) fab.dataPtr(), FFTX_DEVICE_FFT_FORWARD);
    fftx_execute_1d(plan, (double*) fab.dataPtr(), (double*) fab2.dataPtr(), FFTX_DEVICE_FFT_INVERSE);

    Gpu::synchronize();
    double t0 = amrex::second();

    amrex::Print() << "    Warm-up: " << t0-t00 << "\n";

    for (int itest = 0; itest < ntests; ++itest) {
        double ta = amrex::second();
        fftx_execute_1d(plan, (double*) fab2.dataPtr(), (double*) fab.dataPtr(), FFTX_DEVICE_FFT_FORWARD);
        fftx_execute_1d(plan, (double*) fab.dataPtr(), (double*) fab2.dataPtr(), FFTX_DEVICE_FFT_INVERSE);
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

        int batch_size = 1;

        {
            ParmParse pp;
            AMREX_D_TERM(pp.query("n_cell_x", n_cell_x);,
                         pp.query("n_cell_y", n_cell_y);,
                         pp.query("n_cell_z", n_cell_z));
            pp.query("batch_size", batch_size);
        }

        amrex::Print() << "\n FFT size: " << n_cell_x << " " << n_cell_y << " " << n_cell_z
                       << "  batch size: " << batch_size
                       << "  # of proc. " << ParallelDescriptor::NProcs() << "\n\n";

        Box domain(IntVect(0),IntVect(n_cell_x-1,n_cell_y-1,n_cell_z-1));
        BoxArray ba = amrex::decompose(domain, ParallelDescriptor::NProcs(), {false,false,true});

        AMREX_ALWAYS_ASSERT(ba.size() == ParallelDescriptor::NProcs());
        DistributionMapping dm = FFT::detail::make_iota_distromap(ba.size());

        for (int ibox = 0; ibox < ParallelDescriptor::NProcs(); ibox++)
          {
            const Box& bx = ba[ibox];
            const int* lo = bx.loVect();
            const int* hi = bx.hiVect();
            amrex::Print() << "box " << ibox << " ["
                           << lo[0] << ":" << hi[0] << ", "
                           << lo[1] << ":" << hi[1] << ", "
                           << lo[2] << ":" << hi[2] << "]\n";
          }

        GpuArray<Real,3> dx{1._rt/Real(n_cell_x), 1._rt/Real(n_cell_y), 1._rt/Real(n_cell_z)};

        cMultiFab mf(ba, dm, batch_size, 0);
        cMultiFab mf2(ba, dm, batch_size, 0);
        auto const& ma = mf.arrays();
        ParallelFor(mf, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            AMREX_D_TERM(Real x = (i+0.5_rt) * dx[0] - 0.5_rt;,
                         Real y = (j+0.5_rt) * dx[1] - 0.5_rt;,
                         Real z = (k+0.5_rt) * dx[2] - 0.5_rt);
            auto tmp = std::exp(-10._rt*
                (AMREX_D_TERM(x*x*1.05_rt, + y*y*0.90_rt, + z*z)));
            for (int n = 0; n < batch_size; ++n) {
                ma[b](i,j,k) = GpuComplex<Real>(tmp + Real(batch_size), 1._rt-tmp);
            }
        });
        Gpu::streamSynchronize();

        auto t_amrex_auto = test_amrex_auto(domain, mf, mf2);
        amrex::Print() << "  amrex fft time: " << t_amrex_auto << "\n\n";

        {
            MultiFab errmf(ba,dm,1,0);
            auto const& errma = errmf.arrays();

            auto scaling = Real(1.0 / domain.d_numPts());

            auto const& ma2 = mf2.const_arrays();
            ParallelFor(mf2, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
            {
                auto err = ma[b](i,j,k) - ma2[b](i,j,k)*scaling;
                errma[b](i,j,k) = amrex::norm(err);
            });

            auto error = errmf.norminf();
            amrex::Print() << "  Expected to be close to zero: " << error << "\n\n";
        }

#ifdef USE_FFTX
        auto t_fftx = test_fftx(domain, mf, mf2);
        amrex::Print() << "  fftx dist    time: " << t_fftx << "\n";
#endif
    }
    amrex::Finalize();
}
