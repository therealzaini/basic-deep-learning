#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
using namespace std;

double dot(vector<double> u, vector<double> v);

class Matrix{
    public:
        vector<int> format;
        vector<vector<double>> matrix;
        Matrix(vector<vector<double>> matrix);
        double get_entry(int i, int j) const;
        void set_entry(double val, int i, int j);
        vector<double> get_row(int i) const;
        vector<double> get_column(int j) const;
        static Matrix zero(int n, int p);
        void T();
        Matrix operator+(const Matrix& B);
        Matrix operator-(const Matrix& B); 
        Matrix operator%(const Matrix& B); //Component-wise matrix multiplication.
        Matrix operator^(const Matrix& B) const; //Matrix multiplication.
        static Matrix randomize(int n, int p, double min_value, double max_value);
};
#endif