#include <mpi/mpi.h>
#include <sstream>
#include <random>
#include <iostream>

#define I_COUNT 12
#define J_COUNT 12
#define PROCESS_COUNT 4
#define I_COUNT_IN_LOCAL_MATRIX I_COUNT/PROCESS_COUNT
#define MAX_GENERATED_VALUE 100.0
#define FLUSH_STDOUT fflush(stdout)
#define REPEAT_COUNT 100
#define epsilon 0.01

//MATRIX
double g_Matrix[I_COUNT][J_COUNT];
double l_Matrix[I_COUNT][J_COUNT];

//MPI
int rank;
int totalProcessesCount;

void printGlobalMatrix()
{
    std::stringstream str;
    for(auto &row : g_Matrix)
    {
        for (auto &value : row)
        {
            str << value << " ";
        }
        str << std::endl;
    }
    str << "---END---" << std::endl;
    std::cout << str.str();
    FLUSH_STDOUT;
}

void printLocalMatrix()
{
    std::stringstream str;
    for(auto &row : l_Matrix)
    {
        for (auto &value : row)
        {
            str << value << " ";
        }
        str << std::endl;
    }
    str << "---END---" << std::endl;
    std::cout << str.str();
    FLUSH_STDOUT;
}
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &totalProcessesCount);
    if(rank == 0)
    {
        //Generate matrix
        std::uniform_real_distribution<double> unif(0.0, MAX_GENERATED_VALUE);
        std::default_random_engine re((unsigned long)time(nullptr));
        for(double (&row)[I_COUNT] : g_Matrix)
        {
            for(double &value : row)
            {
                value = unif(re);
            }
        }
        printGlobalMatrix();
    }
    MPI_Bcast(g_Matrix, I_COUNT*J_COUNT, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double leftValue, rightValue, upValue, downValue, diff = 0.0, sqrtdiff = epsilon;
    int k = 0;
    for(; k < REPEAT_COUNT && sqrtdiff >= epsilon; k++)
    {
        for (int i = rank * I_COUNT_IN_LOCAL_MATRIX; i < (rank + 1) * I_COUNT_IN_LOCAL_MATRIX; i++)
        {
            for (int j = 0; j < J_COUNT; j++)
            {
                //Find leftValue & rightValue
                if (j == 0)
                {
                    leftValue = g_Matrix[i][J_COUNT - 1];
                    rightValue = g_Matrix[i][1];
                } else if (j == J_COUNT - 1)
                {
                    leftValue = g_Matrix[i][j - 1];
                    rightValue = g_Matrix[i][0];
                } else
                {
                    leftValue = g_Matrix[i][j - 1];
                    rightValue = g_Matrix[i][j + 1];
                }
                //Find upValue & downValue
                if (i == 0)
                {
                    upValue = g_Matrix[I_COUNT - 1][j];
                    downValue = g_Matrix[i + 1][j];
                } else if (i == I_COUNT - 1)
                {
                    upValue = g_Matrix[i - 1][j];
                    downValue = g_Matrix[0][j];
                } else
                {
                    upValue = g_Matrix[i - 1][j];
                    downValue = g_Matrix[i + 1][i];
                }
                l_Matrix[i][j] = (leftValue + rightValue + upValue + downValue) / 4;
                diff += (l_Matrix[i][j] - g_Matrix[i][j])*(l_Matrix[i][j] - g_Matrix[i][j]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(l_Matrix, g_Matrix, I_COUNT * J_COUNT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        sqrtdiff = sqrt(diff);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    {
        printf("Found a right solution. Epsilon(Difference) = %f, K = %d\n", sqrtdiff, k);
        printGlobalMatrix();
    }
    MPI_Finalize();
    return 0;
}