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

    // Compute A2
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
    int i, j;
    double** matrix;
    int lmax = 5;
    int diff = 0;
    int N = (lmax + 1) * (lmax + 1);

    // Log it
    cout << "Testing rotation matrix R... ";

    // Initialize an empty matrix
    matrix = new double*[N];
    for (i=0; i<N; i++) {
        matrix[i] = new double[N];
        for (j=0; j<N; j++)
            matrix[i][j] = 0;
    }

    // Let's do some basic rotations
    double u[3];
    double theta = M_PI / 2.;

    // Rotate by PI/2 about x
    u[0] = 1; u[1] = 0; u[2] = 0;
    R(lmax, u, theta, matrix);
    for (i=0; i<N; i++) {
        diff += mapdiff(N, matrix[i], TEST_RX[i]);
    }

    // Rotate by PI/2 about y
    u[0] = 0; u[1] = 1; u[2] = 0;
    R(lmax, u, theta, matrix);
    for (i=0; i<N; i++) {
        diff += mapdiff(N, matrix[i], TEST_RY[i]);
    }

    // Rotate by PI/2 about z
    u[0] = 0; u[1] = 0; u[2] = 1;
    R(lmax, u, theta, matrix);
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
Benchmark test for r()

*/
int test_r() {
    double* vector;
    int lmax = 5;
    int diff = 0;
    int N = (lmax + 1) * (lmax + 1);

    // Log it
    cout << "Testing phase curve solution vector r... ";

    // Initialize an empty vector
    vector = new double[N];

    // Compute the phase curve solution vector
    r(lmax, vector);

    // Compare to benchmark
    diff = mapdiff(N, vector, TEST_LITTLER);

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
Run all tests.

*/
int main(){
    int diff = test_A1() || test_A2() || test_A() || test_R() || test_r();
    if (diff == 0)
        cout << "All tests passed." << endl;
    else
        cout << "One or more tests failed." << endl;
    return diff;
}
