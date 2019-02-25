/**

*/
template <
    typename U=S, 
    typename=IsEmitted<U>
>
inline void computeLinearIntensityModelInternal (
    const Scalar& theta, 
    const Vector<Scalar>& x, 
    const Vector<Scalar>& y,
    RowMatrix<Scalar>& A
) {

    // \todo

}

/**

*/
template <
    typename U=S, 
    typename=IsReflected<U>
>
inline void computeLinearIntensityModelInternal (
    const Scalar& theta, 
    const Vector<Scalar>& x, 
    const Vector<Scalar>& y,
    const UnitVector<Scalar>& source,
    RowMatrix<Scalar>& A
) {

    // \todo

}