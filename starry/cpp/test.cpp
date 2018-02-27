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
Run all tests.

*/
int main(){
    int diff = test_A1() || test_A2() || test_A();
    if (diff == 0)
        cout << "All tests passed." << endl;
    else
        cout << "One or more tests failed." << endl;
    return diff;
}
