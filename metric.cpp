#include <immintrin.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#define MATRIX_SIZE 2048
#define NUM_THREAD 2
#define VECTOR_WIDTH 16

// 定义线程范围结构体
typedef struct
{
    int start;
    int end;
} ThreadRange;

// 定义线程池结构体
typedef struct
{
    ThreadRange *ranges;
    int num_ranges;
} ThreadPool;

float matA[MATRIX_SIZE][MATRIX_SIZE];
float matB[MATRIX_SIZE][MATRIX_SIZE];
float *matC;
int step = 0;

// 初始化线程池，将计算任务分配给不同的线程
void initializeThreadPool(ThreadPool *pool, int num_threads)
{
    pool->num_ranges = num_threads;
    pool->ranges = (ThreadRange *)malloc(num_threads * sizeof(ThreadRange));

    int chunk_size = MATRIX_SIZE / num_threads;
    for (int i = 0; i < num_threads; ++i)
    {
        pool->ranges[i].start = i * chunk_size;
        pool->ranges[i].end = (i + 1) * chunk_size;
    }
}

// 销毁线程池，释放内存
void destroyThreadPool(ThreadPool *pool)
{
    free(pool->ranges);
}

// AVX-512矩阵乘法运算，该函数会被多个线程同时调用
void *multiplicationAVX512(void *args)
{
    ThreadPool *pool = (ThreadPool *)args;

    // 使用原子操作获取并增加step，以确定当前线程的工作范围
    int thread = __sync_fetch_and_add(&step, 1);

    // 判断是否有多余的线程，否则退出
    if (thread >= pool->num_ranges)
    {
        return NULL;
    }

    // 循环遍历当前线程分配的行范围
    for (int i = pool->ranges[thread].start; i < pool->ranges[thread].end; ++i)
    {
        // 初始化结果矩阵的当前行为0
        for (int j = 0; j < MATRIX_SIZE; ++j)
        {
            matC[i * MATRIX_SIZE + j] = 0.0f;
        }

        // 循环遍历矩阵B的列
        for (int j = 0; j < MATRIX_SIZE; ++j)
        {
            // 为矩阵A的当前元素创建一个512位向量
            __m512 vecA = _mm512_set1_ps(matA[i][j]);

            // 循环遍历矩阵B的行，以向量宽度为步长
            for (int k = 0; k < MATRIX_SIZE; k += VECTOR_WIDTH)
            {
                // 从内存中加载矩阵B的向量
                __m512 vecB = _mm512_loadu_ps(&matB[j][k]);
                // 从内存中加载结果矩阵的向量
                __m512 vecC = _mm512_loadu_ps(&matC[i * MATRIX_SIZE + k]);
                // 使用FMA指令执行矩阵乘法累加操作
                vecC = _mm512_fmadd_ps(vecA, vecB, vecC);
                // 将结果存回内存
                _mm512_storeu_ps(&matC[i * MATRIX_SIZE + k], vecC);
            }
        }
    }
    return NULL;
}

// 初始化矩阵
void createMatrix()
{
    for (int i = 0; i < MATRIX_SIZE; ++i)
    {
        for (int j = 0; j < MATRIX_SIZE; ++j)
        {
            matA[i][j] = i + j * 2;
            matB[i][j] = i * 2 + j;
        }
    }
}

// 打印矩阵
void printMatrix()
{
    if (MATRIX_SIZE <= 16)
    {
        printf("Matrix A\n");
        for (int i = 0; i < MATRIX_SIZE; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                printf("%f ", matA[i][j]);
            }
            printf("\n");
        }

        printf("\nMatrix B\n");
        for (int i = 0; i < MATRIX_SIZE; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                printf("%f ", matB[i][j]);
            }
            printf("\n");
        }

        printf("\nMultiplication Result (Matrix C)\n");
        for (int i = 0; i < MATRIX_SIZE; i++)
        {
            for (int j = 0; j < MATRIX_SIZE; j++)
            {
                printf("%f ", matC[i * MATRIX_SIZE + j]);
            }
            printf("\n");
        }
    }
}

// 主函数
int main()
{
    pthread_t threads[NUM_THREAD];
    ThreadPool threadPool;
    clock_t start, end;
    int prot = PROT_READ | PROT_WRITE;
    int flags = MAP_SHARED | MAP_ANONYMOUS;
    int fd = -1;
    off_t offset = 0;

    // 初始化矩阵
    createMatrix();
    // 初始化线程池
    initializeThreadPool(&threadPool, NUM_THREAD);

    // 创建共享内存映射区域并设置保护属性
    matC = mmap(NULL, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, prot, flags, fd, offset);
    if (matC == MAP_FAILED)
    {
        perror("mmap");
        exit(1);
    }

    // 记录开始时间
    start = clock();

    // 创建线程并执行AVX-512矩阵乘法
    for (int i = 0; i < NUM_THREAD; ++i)
    {
        pthread_create(&threads[i], NULL, multiplicationAVX512, (void *)&threadPool);
    }

    // 等待线程结束
    for (int i = 0; i < NUM_THREAD; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    // 记录结束时间
    end = clock();

    printf("\n\n使用的线程数 -> %d\n", NUM_THREAD);
    printf("\n矩阵大小 -> %d\n", MATRIX_SIZE);
    printf("\n程序运行时间(毫秒) -> %f\n\n", (float)(end - start) * 1000 / CLOCKS_PER_SEC);
    // 释放共享内存映射区域
    munmap(matC, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    return 0;
}

