#include "Matrix.h"
#include <vector>
#include <ctime>
#include <cstdlib>
#include <algorithm>
using namespace std;

double dot(vector<double> u, vector<double> v){
    double dot_product = 0;
    for(int i = 0; i < u.size(); i++){
        dot_product += u[i]*v[i];
    }
    return dot_product;
}

Matrix::Matrix(vector<vector<double>> inpt_matrix){
    int n = inpt_matrix.size();

    int pmax = 0;
    for(int i = 0; i < n; i++){
        if(inpt_matrix[i].size() > pmax) {
            pmax = inpt_matrix[i].size();
        }
    }
    
    format = {n, pmax};
    matrix = inpt_matrix;
}

double Matrix::get_entry(int i, int j) const {
    return matrix[i-1][j-1];
}

void Matrix::set_entry(double val, int i, int j){
    matrix[i-1][j-1] = val;
}

vector<double> Matrix::get_row(int i) const {
    return matrix[i-1];
}

vector<double> Matrix::get_column(int j) const {
    vector<double> column;
    for(vector<double> row : matrix){
        column.push_back(row[j-1]);
    }
    return column;
}

Matrix Matrix::zero(int n, int p){
    vector<vector<double>> M;
    for(int i = 0; i < n; i++){
        vector<double> row;
        for(int j = 0; j < p; j++){
            row.push_back(0);
        }
        M.push_back(row);
    }
    return Matrix(M);
}

void Matrix::T(){
    vector<vector<double>> copy_matrix = matrix;
    int original_format[2] = {format[0], format[1]};
    matrix = Matrix::zero(original_format[1],original_format[0]).matrix;
    format = {original_format[1], original_format[0]};
    for(int i = 0; i < format[0]; i++){
        for(int j = 0; j < format[1]; j++){
            matrix[i][j] = copy_matrix[j][i];
        }
    }
}

Matrix Matrix::operator+(const Matrix& B){
    vector<vector<double>> matrix_sum;
    for(int i = 0; i < format[0]; i++){
        vector<double> row;
        for(int j = 0; j < format[1]; j++){
            row.push_back(matrix[i][j] + B.matrix[i][j]);
        }
        matrix_sum.push_back(row);
    }
    return Matrix(matrix_sum);
}

Matrix Matrix::operator-(const Matrix& B){
    vector<vector<double>> matrix_diff;
    for(int i = 0; i < format[0]; i++){
        vector<double> row;
        for(int j = 0; j < format[1]; j++){
            row.push_back(matrix[i][j] - B.matrix[i][j]);
        }
        matrix_diff.push_back(row);
    }
    return Matrix(matrix_diff);
}

Matrix Matrix::operator%(const Matrix& B){
    vector<vector<double>> matrix_cwise;
    for(int i = 0; i < format[0]; i++){
        vector<double> row;
        for(int j = 0; j < format[1]; j++){
            row.push_back(matrix[i][j] * B.matrix[i][j]);
        }
        matrix_cwise.push_back(row);
    }
    return Matrix(matrix_cwise);
}

Matrix Matrix::operator^(const Matrix& B) const {
    Matrix matrix_product = Matrix::zero(format[0], B.format[1]);
    for(int i = 1; i < format[0] + 1; i++){
        for(int j = 1; j < B.format[1] + 1; j++){
            double value = dot(get_row(i), B.get_column(j));
            matrix_product.set_entry(value, i, j);
        }
    }
    return matrix_product;
}

Matrix Matrix::randomize(int n, int p, double min_value, double max_value){
    Matrix random_matrix = Matrix::zero(n, p);
    srand(time(0));
    for(int i = 1; i < n + 1 ; i++){
        for(int j = 1; j < p + 1; j++){
            double t = static_cast<double>(rand()) / RAND_MAX;
            random_matrix.set_entry(min_value * (1 - t) + t * max_value, i, j);
        }
    }
    return random_matrix;
}