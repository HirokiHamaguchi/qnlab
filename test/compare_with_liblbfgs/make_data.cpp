#include <lbfgs.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>

static std::string global_folder_name;

class RosenbrockFunction {
 public:
  RosenbrockFunction() {}

  int run(int n) {
    lbfgsfloatval_t fx;
    lbfgsfloatval_t* x = lbfgs_malloc(n);
    for (int i = 0; i < n; i += 2) {
      x[i] = -1.2;
      x[i + 1] = 1.0;
    }
    int ret = lbfgs(n, x, &fx, _evaluate, _progress, this, NULL);
    return ret;
  }

 protected:
  static lbfgsfloatval_t _evaluate(void* instance, const lbfgsfloatval_t* x,
                                   lbfgsfloatval_t* g, const int n,
                                   const lbfgsfloatval_t step) {
    lbfgsfloatval_t fx = 0.0;
    for (int i = 0; i < n; ++i) g[i] = 0;
    for (int i = 0; i < n; i += 2) {
      lbfgsfloatval_t t1 = 1.0 - x[i];
      lbfgsfloatval_t t2 = 10.0 * (x[i + 1] - x[i] * x[i]);
      fx += t1 * t1 + t2 * t2;
      g[i + 1] = 20.0 * t2;
      g[i] = -2.0 * (x[i] * g[i + 1] + t1);
    }
    return fx;
  }

  static int _progress(void* instance, const lbfgsfloatval_t* x,
                       const lbfgsfloatval_t* g, lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls) {
    FILE* fp = fopen(
        (global_folder_name + "/RosenbrockFunction_results.txt").c_str(), "a");
    if (fp) {
      fprintf(fp, "%f\n", gnorm);
      fclose(fp);
    }
    return 0;
  }
};

//
// Dixon-Price Function
// f(x) = (x0-1)^2 + sum_{i=1}^{n-1} i*(2*x[i]^2 - x[i-1])^2
//
class DixonPriceFunction {
 public:
  DixonPriceFunction() {}

  int run(int n) {
    lbfgsfloatval_t fx;
    lbfgsfloatval_t* x = lbfgs_malloc(n);
    for (int i = 0; i < n; i++) x[i] = 0.5;
    int ret = lbfgs(n, x, &fx, _evaluate, _progress, this, NULL);
    return ret;
  }

 protected:
  static lbfgsfloatval_t _evaluate(void* instance, const lbfgsfloatval_t* x,
                                   lbfgsfloatval_t* g, int n,
                                   lbfgsfloatval_t step) {
    lbfgsfloatval_t fx = 0.0;
    for (int i = 0; i < n; ++i) g[i] = 0;
    fx += (x[0] - 1.0) * (x[0] - 1.0);
    g[0] = 2.0 * (x[0] - 1.0);
    for (int i = 1; i < n; i++) {
      lbfgsfloatval_t temp = 2.0 * x[i] * x[i] - x[i - 1];
      fx += i * temp * temp;
      g[i] += 4.0 * i * x[i] * temp;
      g[i - 1] -= 2.0 * i * temp;
    }
    return fx;
  }

  static int _progress(void* instance, const lbfgsfloatval_t* x,
                       const lbfgsfloatval_t* g, lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls) {
    FILE* fp = fopen(
        (global_folder_name + "/DixonPriceFunction_results.txt").c_str(), "a");
    if (fp) {
      fprintf(fp, "%f\n", gnorm);
      fclose(fp);
    }
    return 0;
  }
};

//
// Powell Function (badly scaled), dimension fixed to 4:
// f(x) = (x0+10*x1)^2 + 5*(x2-x3)^2 + (x1-2*x2)^4 + 10*(x0-x3)^4
//
class PowellFunction {
 public:
  PowellFunction() {}

  int run(int n) {
    if (n != 4) {
      printf("ERROR: Powell function requires n==4.\n");
      return 1;
    }
    lbfgsfloatval_t fx;
    lbfgsfloatval_t* x = lbfgs_malloc(n);
    x[0] = 3.0;
    x[1] = -1.0;
    x[2] = 0.0;
    x[3] = 1.0;
    int ret = lbfgs(n, x, &fx, _evaluate, _progress, this, NULL);
    return ret;
  }

 protected:
  static lbfgsfloatval_t _evaluate(void* instance, const lbfgsfloatval_t* x,
                                   lbfgsfloatval_t* g, int n,
                                   lbfgsfloatval_t step) {
    lbfgsfloatval_t A = x[0] + 10.0 * x[1];
    lbfgsfloatval_t B = x[2] - x[3];
    lbfgsfloatval_t C = x[1] - 2.0 * x[2];
    lbfgsfloatval_t D = x[0] - x[3];
    lbfgsfloatval_t fx = A * A + 5.0 * B * B + pow(C, 4) + 10.0 * pow(D, 4);
    for (int i = 0; i < n; ++i) g[i] = 0;
    g[0] += 2.0 * A;
    g[1] += 20.0 * A;
    g[2] += 10.0 * B;
    g[3] -= 10.0 * B;
    lbfgsfloatval_t C3 = 4.0 * pow(C, 3);
    g[1] += C3;
    g[2] -= 2.0 * C3;
    lbfgsfloatval_t D3 = 40.0 * pow(D, 3);
    g[0] += D3;
    g[3] -= D3;
    return fx;
  }

  static int _progress(void* instance, const lbfgsfloatval_t* x,
                       const lbfgsfloatval_t* g, lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls) {
    FILE* fp = fopen(
        (global_folder_name + "/PowellFunction_results.txt").c_str(), "a");
    if (fp) {
      fprintf(fp, "%f\n", gnorm);
      fclose(fp);
    }
    return 0;
  }
};

//
// Zakharov Function
// f(x) = sum_{i=0}^{n-1} x[i]^2 + (sum_{i=0}^{n-1} 0.5*(i+1)*x[i])^2 +
// (sum_{i=0}^{n-1} 0.5*(i+1)*x[i])^4
//
class ZakharovFunction {
 public:
  ZakharovFunction() {}

  int run(int n) {
    lbfgsfloatval_t fx;
    lbfgsfloatval_t* x = lbfgs_malloc(n);
    for (int i = 0; i < n; i++) x[i] = 1.0;
    int ret = lbfgs(n, x, &fx, _evaluate, _progress, this, NULL);
    return ret;
  }

 protected:
  static lbfgsfloatval_t _evaluate(void* instance, const lbfgsfloatval_t* x,
                                   lbfgsfloatval_t* g, int n,
                                   lbfgsfloatval_t step) {
    lbfgsfloatval_t fx = 0.0, s = 0.0;
    for (int i = 0; i < n; i++) {
      fx += x[i] * x[i];
      s += 0.5 * (i + 1) * x[i];
      g[i] = 0;
    }
    fx += s * s + s * s * s * s;
    for (int i = 0; i < n; i++) {
      lbfgsfloatval_t ai = 0.5 * (i + 1);
      g[i] = 2.0 * x[i] + (2.0 * s + 4.0 * s * s * s) * ai;
    }
    return fx;
  }

  static int _progress(void* instance, const lbfgsfloatval_t* x,
                       const lbfgsfloatval_t* g, lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls) {
    FILE* fp = fopen(
        (global_folder_name + "/ZakharovFunction_results.txt").c_str(), "a");
    if (fp) {
      fprintf(fp, "%f\n", gnorm);
      fclose(fp);
    }
    return 0;
  }
};

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <folder_name>\n", argv[0]);
    return 1;
  }
  const std::string folder_name = argv[1];
  global_folder_name = folder_name;

  {
    RosenbrockFunction func;
    func.run(100);
  }
  {
    DixonPriceFunction func;
    func.run(100);
  }
  {
    PowellFunction func;
    func.run(4);
  }
  {
    ZakharovFunction func;
    func.run(100);
  }
  return 0;
}