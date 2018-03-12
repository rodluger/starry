#include "starry.h"
#include "test.h"

/**
Test whether two maps are equal up to order N.

*/
int mapdiff(int N, double y1[N], double y2[N], double tol=1e-8) {
    int diff = 0;
    int i;
    for (i = 0; i < N; i++) {
        if (abs(y1[i] - y2[i]) > tol) {
            if (diff == 0)
                cout << endl;
            cout << "Error at index " << i << ":  " << y1[i] << "   " << y2[i] << endl;
            diff++;
        }
    }
    return diff;
}


/**
Benchmark test for A1()

*/
int test_A1() {
    int i, j;
    double** matrix;
    int lmax = 5;
    int N = (lmax + 1) * (lmax + 1);

    // Log it
    cout << "Testing change of basis matrix A1... ";

    // Initialize an empty matrix
    matrix = new double*[N];
    for (i=0; i<N; i++) {
        matrix[i] = new double[N];
        for (j=0; j<N; j++)
            matrix[i][j] = 0;
    }

    // Compute A1
    A1(lmax, matrix);

    // Compare to the benchmark
    int diff = 0;
    for (i=0; i<N; i++) {
        diff += mapdiff(N, matrix[i], TEST_A1[i]);
    }

    // Log it
    if (diff == 0)
        cout << "OK" << endl;
    else
        cout << "ERROR" << endl;

    // Free
    for(i = 0; i<N; i++)
        delete [] matrix[i];
    delete [] matrix;

    // Return zero if we're all good
    return diff;

}

/**
Benchmark test for A2()

*/
int test_A2() {
    int i, j;
    double** matrix;
    int lmax = 3;
    int N = (lmax + 1) * (lmax + 1);

    // Log it
    cout << "Testing change of basis matrix A2... ";

    // Initialize an empty matrix
    matrix = new double*[N];
    for (i=0; i<N; i++) {
        matrix[i] = new double[N];
        for (j=0; j<N; j++)
            matrix[i][j] = 0;
    }

    // Compute A2
    A2(lmax, matrix);

    // Compare to the benchmark
    int diff = 0;
    for (i=0; i<N; i++) {
        diff += mapdiff(N, matrix[i], TEST_A2[i]);
    }

    // Log it
    if (diff == 0)
        cout << "OK" << endl;
    else
        cout << "ERROR" << endl;

    // Free
    for(i = 0; i<N; i++)
        delete [] matrix[i];
    delete [] matrix;

    // Return zero if we're all good
    return diff;

}

/**
Benchmark test for A()

*/
int test_A() {
    int i, j;
    double** matrix;
    int lmax = 3;
    int N = (lmax + 1) * (lmax + 1);

    // Log it
    cout << "Testing change of basis matrix A... ";

    // Initialize an empty matrix
    matrix = new double*[N];
    for (i=0; i<N; i++) {
        matrix[i] = new double[N];
        for (j=0; j<N; j++)
            matrix[i][j] = 0;
    }

    // Compute A
    A(lmax, matrix);

    // Compare to the benchmark
    int diff = 0;
    for (i=0; i<N; i++) {
        diff += mapdiff(N, matrix[i], TEST_A[i]);
    }

    // Log it
    if (diff == 0)
        cout << "OK" << endl;
    else
        cout << "ERROR" << endl;

    // Free
    for(i = 0; i<N; i++)
        delete [] matrix[i];
    delete [] matrix;

    // Return zero if we're all good
    return diff;

}

/**
Benchmark test for R()

*/
int test_R() {
    int i;
    double** matrix;
    int lmax = 5;
    int diff = 0;
    int N = (lmax + 1) * (lmax + 1);

    // Log it
    cout << "Testing rotation matrix R... ";

    // Initialize an empty matrix
    matrix = new double*[N];
    for (i=0; i<N; i++)
        matrix[i] = new double[N];

    // Let's do some basic rotations
    double u[3];
    double costheta = 0;
    double sintheta = 1;

    // Rotate by PI/2 about x
    u[0] = 1; u[1] = 0; u[2] = 0;
    R(lmax, u, costheta, sintheta, matrix);
    for (i=0; i<N; i++) {
        diff += mapdiff(N, matrix[i], TEST_RX[i]);
    }

    // Rotate by PI/2 about y
    u[0] = 0; u[1] = 1; u[2] = 0;
    R(lmax, u, costheta, sintheta, matrix);
    for (i=0; i<N; i++) {
        diff += mapdiff(N, matrix[i], TEST_RY[i]);
    }

    // Rotate by PI/2 about z
    u[0] = 0; u[1] = 0; u[2] = 1;
    R(lmax, u, costheta, sintheta, matrix);
    for (i=0; i<N; i++) {
        diff += mapdiff(N, matrix[i], TEST_RZ[i]);
    }

    // Log it
    if (diff == 0)
        cout << "OK" << endl;
    else
        cout << "ERROR" << endl;

    // Free
    for(i = 0; i<N; i++)
        delete [] matrix[i];
    delete [] matrix;

    // Return zero if we're all good
    return diff;

}

/**
Benchmark test for rT()

*/
int test_rT() {
    double* vector;
    int lmax = 5;
    int diff = 0;
    int N = (lmax + 1) * (lmax + 1);

    // Log it
    cout << "Testing phase curve solution vector r... ";

    // Initialize an empty vector
    vector = new double[N];

    // Compute the phase curve solution vector
    rT(lmax, vector);

    // Compare to benchmark
    diff = mapdiff(N, vector, TEST_RT);

    // Log it
    if (diff == 0)
        cout << "OK" << endl;
    else
        cout << "ERROR" << endl;

    // Free
    delete [] vector;

    // Return zero if we're all good
    return diff;

}

/**
Benchmark test for sT()

*/
int test_sT() {
    double* vector;
    int lmax = 5;
    int diff = 0;
    int N = (lmax + 1) * (lmax + 1);

    // Log it
    cout << "Testing occultation solution vector s... ";

    // Initialize an empty vector
    vector = new double[N];

    // Compute the phase curve solution vector
    // and compare to benchmark for a few different
    // occultation parameters

    // 1. Small occultor
    sT(lmax, 0.5, 0.3, vector);
    diff += mapdiff(N, vector, TEST_ST53);
    // 2. Large occultor
    sT(lmax, 0.9, 1.5, vector);
    diff += mapdiff(N, vector, TEST_ST915);
    // 3. Zero impact parameter
    sT(lmax, 0, 0.5, vector);
    diff += mapdiff(N, vector, TEST_ST05);
    // 4. Singular k = 1 case
    sT(lmax, 0.5, 0.5, vector);
    diff += mapdiff(N, vector, TEST_ST55);
    // 5. Test code as k --> 1. Currently numerically unstable (TODO)
    // 6. Test code as k --> inf. Currently numerically unstable (TODO)
    // 7. Test other k = 1 cases (r < 0.5, r > 0.5) (TODO)

    // Log it
    if (diff == 0)
        cout << "OK" << endl;
    else
        cout << "ERROR" << endl;

    // Free
    delete [] vector;

    // Return zero if we're all good
    return diff;

}

/**
Benchmark test for main starry routines

*/
int test_starry() {
    int diff = 0;
    int lmax = 5;
    int NT = 50;
    CONSTANTS C;
    init_constants(lmax, &C);
    double yhat[3] = {0., 1., 0.};
    double* y = new double[C.N];
    for (int i=0; i<C.N; i++) y[i] = 0;
    double r;
    double* theta = new double[NT];
    double* x0 = new double[NT];
    double* y0 = new double[NT];
    double* result = new double[NT];

    // Log it
    cout << "Testing main starry routine... ";

    // Benchmarked occultation params
    // y = Y_{0,0} + Y_{1,1} + Y_{2,-2} + Y_{3,-1} + Y_{4,-3} + Y_{5,0}
    y[0] = 1;
    y[3] = 1;
    y[4] = 1;
    y[11] = 1;
    y[17] = 1;
    y[30] = 1;
    r = 0.3;
    for (int i=0; i<NT; i++) {
        x0[i] = 2. * i / (NT - 1.) - 1;
        y0[i] = 4. * i / (NT - 1.) - 2;
        theta[i] = 0.5 * M_PI * i / (NT - 1.);
    }

    // Compute the light curve
    flux(NT, y, yhat, theta, x0, y0, r, &C, result);

    // Compare to benchmark
    diff += mapdiff(NT, result, TEST_LC1);

    // Log it
    if (diff == 0)
        cout << "OK" << endl;
    else
        cout << "ERROR" << endl;

    // Free
    free_constants(lmax, &C);
    delete [] y;
    delete [] theta;
    delete [] x0;
    delete [] y0;
    delete [] result;

    return diff;
}

/**
Benchmark test for render()
Let's render the Y_{1,-1} = sqrt(3/(4pi)) * y spherical harmonic.

*/
int test_render() {
    int lmax = 1;
    int diff = 0;
    int res = 10;
    double yhat[3] = {0., 1., 0.};
    double* x = new double[res];
    double norm = sqrt(3 / (4 * M_PI));
    double theta = 0;
    int i, j;
    CONSTANTS C;
    init_constants(lmax, &C);
    double* y = new double[C.N];
    y[0] = 0;
    y[1] = 1;
    y[2] = 0;
    y[3] = 0;
    double** map = new double*[res];
    for (i=0; i<res; i++)
        map[i] = new double[res];

    // Log it
    cout << "Testing the map rendering function... ";

    // Render the map
    render(y, yhat, theta, &C, res, map);

    // Check it
    for (i=0; i<res; i++)
        x[i] = -1. + 2. * i / (res - 1.);
    for (i=0; i<res; i++) {
        for (j=0; j<res; j++) {
            if (x[i] * x[i] + x[j] * x[j] < 1) {
                if (abs(map[j][i] - norm * x[j]) > 1e-8) {
                    diff++;
                }
            }
        }
    }

    // Log it
    if (diff == 0)
        cout << "OK" << endl;
    else
        cout << "ERROR" << endl;

    // Free
    free_constants(lmax, &C);
    for (i=0; i<res; i++)
        delete [] map[i];
    delete [] map;
    delete [] x;
    delete [] y;

    return diff;

}


/**
Run all tests.

*/
int main(){
    int diff = test_A1() || test_A2() || test_A() ||
               test_R() || test_rT() || test_sT() ||
               test_starry() || test_render();
    if (diff == 0)
        cout << "All tests passed." << endl;
    else
        cout << "One or more tests failed." << endl;
    return diff;
}
