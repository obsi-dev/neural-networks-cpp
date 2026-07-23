#include "matrix.hpp"

Matrix *MatrixCreate(Arena *arena, int rows, int cols)
{
    Matrix *mat = PUSH_STRUCT(arena, Matrix);
    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, float, (ull)(rows * cols));

    return mat;
}

Matrix *ReLU(Matrix *a)
{
    int rows = a->rows, cols = a->cols;
    Matrix *result = new Matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++)
    {
        result->data[i] = a->data[i] > 0 ? a->data[i] : 0.0;
    }
    return result;
}

double power(double base, long long exp)
{
    if (exp == 0)
        return 1.0;

    bool isNegative = exp < 0;

    ull exponent = isNegative ? -exp : exp;

    double result = 1.0;

    while (exponent > 0)
    {
        if (exponent & 1)
        {
            result *= base;
        }

        base *= base;
        exponent >>= 1;
    }

    return isNegative ? 1 / result : result;
}

Matrix *MatrixSoftmax(Matrix *a)
{
    int rows = a->rows, cols = a->cols;
    Matrix *result = new Matrix(rows, cols);

    double temp = 0.0;

    for (int i = 0; i < rows * cols; i++)
    {
        result->data[i] = expf(a->data[i]);
        temp += result->data[i];
    }

    for (int i = 0; i < rows * cols; i++)
    {
        result->data[i] /= temp;
    }

    return result;
}

Matrix* MatrixCrossEntropy(const Matrix* p, const Matrix* q) {
    if(p->rows != q->rows || p->cols != q->cols) {
        cout << "Matrix Cross Entropy: Mismatch of matrix dimensions, exiting...";
        return nullptr;    
    }  

    Matrix* result = new Matrix(p->rows, p->cols);

    ull size = (ull)(p->rows * p->cols);

    for(int i = 0; i < size; i++) {
        result->data[i] = p->data[i] == 0.0 ? 0.0 : -p->data[i] * log(q->data[i]);
    }

    return result;
}

Matrix *MatrixSum(Matrix *a, Matrix *b)
{
    if (a->rows != b->rows || a->cols != b->cols)
        return nullptr;

    Matrix *result = new Matrix(a->rows, a->cols);

    for (int i = 0; i < a->rows * a->cols; i++)
    {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

Matrix *MatrixTranspose(Matrix *a)
{
    Matrix *result = new Matrix(a->cols, a->rows);
    for (int r = 0; r < a->rows; r++)
    {
        for (int c = 0; c < a->cols; c++)
        {
            result->data[r * result->cols + c] = a->data[c * a->cols + r];
        }
    }
    return result;
}

Matrix *MatrixMult(Matrix *a, Matrix *b)
{
    if (a->cols != b->rows)
        return nullptr;

    Matrix *result = new Matrix(a->rows, b->cols);

    int a_rows = a->rows, a_cols = a->cols, b_cols = b->cols;

    for (int i = 0; i < a_rows; i++)
    {
        for (int j = 0; j < b_cols; j++)
        {
            for (int k = 0; k < a_cols; k++)
            {
                result->data[i * b_cols + j] += a->data[i * a_cols + k] * b->data[k * b_cols + j];
            }
        }
    }

    return result;
}

Matrix *MatrixLoad(Arena *arena, ull rows, ull cols, const string filepath)
{
    Matrix* mat = MatrixCreate(arena, rows, cols);

    ifstream matfile(filepath, ios::binary);

    if(!matfile.is_open()) {
        cout << "Failed to open file: " << filepath << " , exiting...";
        exit(1);
    }

    matfile.seekg(0, ios::end);
    ull fileSize = (ull)(matfile.tellg());
    fileSize = min(fileSize, sizeof(float) * rows * cols);

    matfile.seekg(0, ios::beg);

    matfile.read((char*)mat->data, fileSize);

    matfile.close();

    return mat;
}
