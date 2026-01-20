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

  static constexpr const char* RESULT_FILE_NAME = "Rosenbrock_results.txt";

  int run(int n) {
    FILE* fp =
        fopen((global_folder_name + "/" + RESULT_FILE_NAME).c_str(), "w");
    if (fp) fclose(fp);
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
    for (int i = 1; i < n; i++) {
      lbfgsfloatval_t temp = x[i] - x[i - 1] * x[i - 1];
      fx += 100.0 * temp * temp + (1.0 - x[i - 1]) * (1.0 - x[i - 1]);
    }
    for (int i = 0; i < n; ++i) g[i] = 0;
    for (int i = 1; i < n - 1; i++) {
      lbfgsfloatval_t xm = x[i];
      lbfgsfloatval_t xm_m1 = x[i - 1];
      lbfgsfloatval_t xm_p1 = x[i + 1];
      g[i] = 200.0 * (xm - xm_m1 * xm_m1) - 400.0 * (xm_p1 - xm * xm) * xm -
             2.0 * (1.0 - xm);
    }
    g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]);
    g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2]);
    return fx;
  }

  static int _progress(void* instance, const lbfgsfloatval_t* x,
                       const lbfgsfloatval_t* g, lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls) {
    FILE* fp =
        fopen((global_folder_name + "/" + RESULT_FILE_NAME).c_str(), "a");
    if (fp) {
      fprintf(fp, "%.10e\n", gnorm);
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

  static constexpr const char* RESULT_FILE_NAME = "DixonPrice_results.txt";

  int run(int n) {
    FILE* fp =
        fopen((global_folder_name + "/" + RESULT_FILE_NAME).c_str(), "w");
    if (fp) fclose(fp);
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
      g[i] += 8.0 * i * x[i] * temp;
      g[i - 1] -= 2.0 * i * temp;
    }
    return fx;
  }

  static int _progress(void* instance, const lbfgsfloatval_t* x,
                       const lbfgsfloatval_t* g, lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls) {
    FILE* fp =
        fopen((global_folder_name + "/" + RESULT_FILE_NAME).c_str(), "a");
    if (fp) {
      fprintf(fp, "%.10e\n", gnorm);
      fclose(fp);
    }
    return 0;
  }
};

//
// Powell Function
// f(x) = (x0+10*x1)^2 + 5*(x2-x3)^2 + (x1-2*x2)^4 + 10*(x0-x3)^4
//
class PowellFunction {
 public:
  PowellFunction() {}

  static constexpr const char* RESULT_FILE_NAME = "Powell_results.txt";

  int run(int n) {
    FILE* fp =
        fopen((global_folder_name + "/" + RESULT_FILE_NAME).c_str(), "w");
    if (fp) fclose(fp);
    if (n % 4 != 0) {
      printf("ERROR: Powell function requires n to be a multiple of 4.\n");
      return 1;
    }
    lbfgsfloatval_t fx;
    lbfgsfloatval_t* x = lbfgs_malloc(n);
    for (int i = 0; i < n / 4; i++) {
      x[4 * i] = 3.0;
      x[4 * i + 1] = -1.0;
      x[4 * i + 2] = 0.0;
      x[4 * i + 3] = 1.0;
    }
    int ret = lbfgs(n, x, &fx, _evaluate, _progress, this, NULL);
    return ret;
  }

 protected:
  static lbfgsfloatval_t _evaluate(void* instance, const lbfgsfloatval_t* x,
                                   lbfgsfloatval_t* g, int n,
                                   lbfgsfloatval_t step) {
    lbfgsfloatval_t fx = 0.0;
    int blocks = n / 4;
    for (int i = 0; i < blocks; i++) {
      int idx = 4 * i;
      lbfgsfloatval_t p = x[idx], q = x[idx + 1], r = x[idx + 2],
                      s = x[idx + 3];
      lbfgsfloatval_t term1 = (p + 10 * q);
      term1 = term1 * term1;
      lbfgsfloatval_t term2 = 5 * (r - s) * (r - s);
      lbfgsfloatval_t term3 = q - 2 * r;
      term3 = pow(term3, 4);
      lbfgsfloatval_t term4 = p - s;
      term4 = 10 * pow(term4, 4);
      fx += term1 + term2 + term3 + term4;
      lbfgsfloatval_t dp = 2 * (p + 10 * q) + 40 * pow(p - s, 3);
      lbfgsfloatval_t dq = 20 * (p + 10 * q) + 4 * pow(q - 2 * r, 3);
      lbfgsfloatval_t dr = 10 * (r - s) - 8 * pow(q - 2 * r, 3);
      lbfgsfloatval_t ds = -10 * (r - s) - 40 * pow(p - s, 3);
      g[idx] = dp;
      g[idx + 1] = dq;
      g[idx + 2] = dr;
      g[idx + 3] = ds;
    }
    return fx;
  }

  static int _progress(void* instance, const lbfgsfloatval_t* x,
                       const lbfgsfloatval_t* g, lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls) {
    FILE* fp =
        fopen((global_folder_name + "/" + RESULT_FILE_NAME).c_str(), "a");
    if (fp) {
      fprintf(fp, "%.10e\n", gnorm);
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

  static constexpr const char* RESULT_FILE_NAME = "Zakharov_results.txt";

  int run(int n) {
    FILE* fp =
        fopen((global_folder_name + "/" + RESULT_FILE_NAME).c_str(), "w");
    if (fp) fclose(fp);
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
    FILE* fp =
        fopen((global_folder_name + "/" + RESULT_FILE_NAME).c_str(), "a");
    if (fp) {
      fprintf(fp, "%.10e\n", gnorm);
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
    func.run(100);
  }
  {
    ZakharovFunction func;
    func.run(100);
  }
  return 0;
}
