// Ground-truth dumper for verifying the Python/torch reimplementation of
// STiC's degradation operators. Includes the *actual* header used by the
// coupled inversion (src/linear_transformations.hpp) and prints the sparse
// operator triplets (row col value) to stdout.
//
// Build:
//   clang++ -O2 -std=c++14 -I../src -I/usr/local/include gt_dump.cc -o gt_dump
//
// Usage:
//   gt_dump psf       nx ny npx npy psf.bin           (float64 npy*npx)
//   gt_dump rebin     rx ry rdx rdy rsx rsy nx ny dx dy sx sy
//   gt_dump destretch nx ny ds.bin                    (float64 2*ny*nx: dx then dy)
//   gt_dump bintrim   nix niy dx dy cx cy nxe nye     (verbatim from coupled_inversion.cc)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include "linear_transformations.hpp"

static std::vector<double> read_bin(const char* fname, size_t n){
  std::vector<double> v(n);
  FILE* f = fopen(fname, "rb");
  if(!f){ fprintf(stderr, "cannot open %s\n", fname); exit(1); }
  if(fread(v.data(), sizeof(double), n, f) != n){
    fprintf(stderr, "short read from %s\n", fname); exit(1);
  }
  fclose(f);
  return v;
}

static void dump_eigen(Eigen::SparseMatrix<double,Eigen::RowMajor,long> const& m){
  for (long k=0; k<m.outerSize(); ++k)
    for (Eigen::SparseMatrix<double,Eigen::RowMajor,long>::InnerIterator it(m,k); it; ++it)
      printf("%ld %ld %.17g\n", (long)it.row(), (long)it.col(), it.value());
}

static void dump_triplets(const long* nz, const double* dat, long nrows, long lnz){
  // nz holds row offsets (after sparse_p* call), dat holds (row, col, val)
  for (long i=0; i<nrows; ++i){
    long const io = nz[i];
    long const in = (i==nrows-1) ? lnz-io : nz[i+1]-io;
    for (long j=0; j<in; ++j)
      printf("%ld %ld %.17g\n", (long)dat[(io+j)*3+0], (long)dat[(io+j)*3+1],
             dat[(io+j)*3+2]);
  }
}

int main(int argc, char** argv){
  if (argc < 2){ fprintf(stderr, "mode required\n"); return 1; }
  std::string mode(argv[1]);
  int const nthread = 1;

  if (mode == "psf"){
    long nx = atol(argv[2]), ny = atol(argv[3]);
    long npx = atol(argv[4]), npy = atol(argv[5]);
    std::vector<double> psf = read_bin(argv[6], npy*npx);
    // Same pre-normalization as coupled_inversion.cc (sum to 1):
    double s = 0; for (double v : psf) s += v;
    for (double& v : psf) v /= s;
    long const cpx = npx/2, cpy = npy/2;
    Eigen::SparseMatrix<double,Eigen::RowMajor,long> tlt;
    lnrtr::get_psf_as_lt<long,double>(nx, ny, npx, npy, cpx, cpy, psf.data(),
                                      nthread, 0.0, 0, true, tlt);
    dump_eigen(tlt);

  } else if (mode == "rebin"){
    long rx = atol(argv[2]), ry = atol(argv[3]);
    double rdx = atof(argv[4]), rdy = atof(argv[5]);
    double rsx = atof(argv[6]), rsy = atof(argv[7]);
    long nx = atol(argv[8]), ny = atol(argv[9]);
    double dx = atof(argv[10]), dy = atof(argv[11]);
    double sx = atof(argv[12]), sy = atof(argv[13]);
    std::vector<long> nz(rx*ry, 0);
    lnrtr::eval_prebin<long,double>(rx, ry, rdx, rdy, rsx, rsy,
                                    nx, ny, dx, dy, sx, sy, nthread, nz.data());
    long lnz = 0; for (long v : nz) lnz += v;
    std::vector<double> dat(lnz*3, 0.0);
    lnrtr::sparse_prebin<long,double>(rx, ry, rdx, rdy, rsx, rsy,
                                      nx, ny, dx, dy, sx, sy, nthread,
                                      nz.data(), dat.data());
    dump_triplets(nz.data(), dat.data(), rx*ry, lnz);

  } else if (mode == "destretch"){
    long nx = atol(argv[2]), ny = atol(argv[3]);
    long const off = nx*ny;
    std::vector<double> ds = read_bin(argv[4], 2*off);
    std::vector<long> nz(off, 0);
    lnrtr::eval_pdestretch<long,double>(nx, ny, &ds[0], &ds[off], nthread, nz.data());
    long lnz = 0; for (long v : nz) lnz += v;
    std::vector<double> dat(lnz*3, 0.0);
    lnrtr::sparse_pdestretch<long,double>(nx, ny, &ds[0], &ds[off], nthread,
                                          nz.data(), dat.data());
    dump_triplets(nz.data(), dat.data(), off, lnz);

  } else if (mode == "bintrim"){
    // Verbatim port of build_lt_sparsematrix, ltid==4 (coupled_inversion.cc)
    int nix = atoi(argv[2]), niy = atoi(argv[3]);
    double dx = atof(argv[4]), dy = atof(argv[5]);
    double cx = atof(argv[6]), cy = atof(argv[7]);
    int nox = int(std::round(atof(argv[8])))-1;
    int noy = int(std::round(atof(argv[9])))-1;
    Eigen::SparseMatrix<double,Eigen::RowMajor,long> m;
    m.resize(long(nox)*noy, long(nix)*niy);
    m.reserve(std::vector<int>(nox*noy, (int)std::ceil(dx+1)*(int)std::ceil(dy+1)));
    for (long s=0; s<m.outerSize(); ++s){
      long itsx = s % nox, itsy = s / nox;
      double botrx = double(itsx)*dx + cx, toprx = botrx + dx;
      double botry = double(itsy)*dy + cy, topry = botry + dy;
      for (long f=0; f<(long(nix)*niy); ++f){
        long itox = f % nix, itoy = f / nix;
        double botox = double(itox), topox = botox + 1;
        double botoy = double(itoy), topoy = botoy + 1;
        if ( (botox>toprx) && (botoy>topry) ){break;}
        if ( (topox<botrx) || (topoy<botry) || (botox>toprx) || (botoy>topry)){continue;}
        double topax = std::min(topox, toprx), topay = std::min(topoy, topry);
        double botax = std::max(botox, botrx), botay = std::max(botoy, botry);
        m.coeffRef(s,f) = (topay - botay) * (topax - botax);
      }
      double norm = m.row(s).sum();
      if (norm != 0){ m.row(s) /= norm; }
    }
    dump_eigen(m);

  } else {
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
  }
  return 0;
}
