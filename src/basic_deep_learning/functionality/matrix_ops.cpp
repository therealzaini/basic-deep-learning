#include <vector>
#include <ctime>
#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

double dot(std::vector<double> u, std::vector<double> v) {
    double dot_product = 0;
    size_t min_size = std::min(u.size(), v.size());
    for (size_t i = 0; i < min_size; i++) {
        dot_product += u[i] * v[i];
    }
    return dot_product;
}

class Matrix {
public:
    std::vector<int> format;
    std::vector<std::vector<double>> matrix;
    
    double get_entry(int i, int j) const;
    void set_entry(double val, int i, int j);
    std::vector<double> get_row(int i) const;
    std::vector<double> get_column(int j) const;
    static Matrix zero(int n, int p);
    static Matrix sum(const Matrix& A, const Matrix& B);
    static Matrix scale(const Matrix& A, double k);
    static Matrix product(const Matrix& A, const Matrix& B);
    static Matrix cwise_product(const Matrix& A, const Matrix& B);
    static Matrix randomize(int n, int p, double min_value, double max_value);
    Matrix(std::vector<std::vector<double>> inpt_matrix);
    
    
    //Python-friendly methods
    py::list get_row_py(int i) const;
    py::list get_column_py(int j) const;
    py::list get_format() const;
    py::list get_matrix() const;
};

Matrix::Matrix(std::vector<std::vector<double>> inpt_matrix) {
    int n = inpt_matrix.size();
    if (n == 0) {
        format = {0, 0};
        return;
    }
    
    int p = 0;
    for (const auto& row : inpt_matrix) {
        if (row.size() > p) p = row.size();
    }
    
    for (auto& row : inpt_matrix) {
        row.resize(p, 0.0);
    }
    matrix = inpt_matrix;
    format = {n, p};
}

double Matrix::get_entry(int i, int j) const {
    if (i < 1 || i > format[0] || j < 1 || j > format[1]) {
        return 0.0;
    }
    return matrix[i-1][j-1];
}

void Matrix::set_entry(double val, int i, int j) {
    if (i < 1 || j < 1) return;
    
    if (i > format[0]) {
        matrix.resize(i, std::vector<double>(format[1], 0.0));
        format[0] = i;
    }
    if (j > format[1]) {
        for (auto& row : matrix) {
            row.resize(j, 0.0);
        }
        format[1] = j;
    }
    
    matrix[i-1][j-1] = val;
}

std::vector<double> Matrix::get_row(int i) const {
    if (i < 1 || i > format[0]) {
        return std::vector<double>();
    }
    return matrix[i-1];
}

std::vector<double> Matrix::get_column(int j) const {
    std::vector<double> column;
    if (j < 1 || j > format[1]) {
        return column;
    }
    
    for (const auto& row : matrix) {
        column.push_back(row[j-1]);
    }
    return column;
}

Matrix Matrix::zero(int n, int p) {
    std::vector<std::vector<double>> M;
    for (int i = 0; i < n; i++) {
        std::vector<double> row(p, 0.0);
        M.push_back(row);
    }
    return Matrix(M);
}

Matrix Matrix::randomize(int n, int p, double min_value, double max_value) {
    Matrix random_matrix = Matrix::zero(n, p);
    std::srand(std::time(0));
    
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= p; j++) {
            double t = static_cast<double>(std::rand()) / RAND_MAX;
            random_matrix.set_entry(min_value + t * (max_value - min_value), i, j);
        }
    }
    return random_matrix;
}

Matrix Matrix::product(const Matrix& A, const Matrix& B) {
    Matrix result = zero(A.format[0], B.format[1]);
    for (int i = 1; i <= A.format[0]; i++) {
        for (int j = 1; j <= B.format[1]; j++) {
            double value = dot(A.get_row(i), B.get_column(j));
            result.set_entry(value, i, j);
        }
    }
    return result;
}

Matrix Matrix::sum(const Matrix& A, const Matrix& B){
    Matrix result = zero(A.format[0], A.format[1]);
    for(int i = 1; i <= A.format[0]; i++){
        for(int j = 1; j <= A.format[1]; j++){
            double a = A.get_entry(i, j);
            double b = B.get_entry(i, j);
            double c = a + b;
            result.set_entry(c, i, j);
        }
    }
    return result;
}

Matrix Matrix::cwise_product(const Matrix& A, const Matrix& B){
    Matrix result = zero(A.format[0], A.format[1]);
    for(int i = 1; i <= A.format[0]; i++){
        for(int j = 1; j <= A.format[1]; j++){
            double a = A.get_entry(i, j);
            double b = B.get_entry(i, j);
            result.set_entry(a * b, i, j);
        }
    }
    return result;
}

Matrix Matrix::scale(const Matrix& A, double k){
    Matrix result = Matrix(A.matrix);
    for(int i = 1; i <= A.format[0]; i++){
        for(int j = 1; j <= A.format[1]; j++){
            result.set_entry(A.get_entry(i, j)*k, i, j);
        }
    }
    return result;
}

// Python-friendly methods
py::list Matrix::get_row_py(int i) const {
    py::list result;
    std::vector<double> row = get_row(i);
    for (double val : row) {
        result.append(val);
    }
    return result;
}

py::list Matrix::get_column_py(int j) const {
    py::list result;
    std::vector<double> col = get_column(j);
    for (double val : col) {
        result.append(val);
    }
    return result;
}

py::list Matrix::get_format() const {
    py::list result;
    for (int val : format) {
        result.append(val);
    }
    return result;
}

py::list Matrix::get_matrix() const {
    py::list result;
    for (const auto& row : matrix) {
        py::list py_row;
        for (double val : row) {
            py_row.append(val);
        }
        result.append(py_row);
    }
    return result;
}

// PyBind11 Module
PYBIND11_MODULE(matrix_ops, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<std::vector<std::vector<double>>>())
        .def("get_entry", &Matrix::get_entry)
        .def("set_entry", &Matrix::set_entry)
        .def("get_row", &Matrix::get_row_py)
        .def("get_column", &Matrix::get_column_py)
        .def("get_format", &Matrix::get_format)
        .def("get_matrix", &Matrix::get_matrix)
        .def_static("zero", &Matrix::zero)
        .def_static("randomize", &Matrix::randomize)
        .def_static("product", &Matrix::product)
        .def_static("cwise_prod", &Matrix::cwise_product)
        .def_static("sum", &Matrix::sum)
        .def_static("scale", &Matrix::scale);
}