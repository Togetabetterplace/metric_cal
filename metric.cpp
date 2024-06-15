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

// �����̷߳�Χ�ṹ��
typedef struct
{
    int start;
    int end;
} ThreadRange;

// �����̳߳ؽṹ��
typedef struct
{
    ThreadRange *ranges;
    int num_ranges;
} ThreadPool;

float matA[MATRIX_SIZE][MATRIX_SIZE];
float matB[MATRIX_SIZE][MATRIX_SIZE];
float *matC;
int step = 0;

// ��ʼ���̳߳أ�����������������ͬ���߳�
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

// �����̳߳أ��ͷ��ڴ�
void destroyThreadPool(ThreadPool *pool)
{
    free(pool->ranges);
}

// AVX-512����˷����㣬�ú����ᱻ����߳�ͬʱ����
void *multiplicationAVX512(void *args)
{
    ThreadPool *pool = (ThreadPool *)args;

    // ʹ��ԭ�Ӳ�����ȡ������step����ȷ����ǰ�̵߳Ĺ�����Χ
    int thread = __sync_fetch_and_add(&step, 1);

    // �ж��Ƿ��ж�����̣߳������˳�
    if (thread >= pool->num_ranges)
    {
        return NULL;
    }

    // ѭ��������ǰ�̷߳�����з�Χ
    for (int i = pool->ranges[thread].start; i < pool->ranges[thread].end; ++i)
    {
        // ��ʼ���������ĵ�ǰ��Ϊ0
        for (int j = 0; j < MATRIX_SIZE; ++j)
        {
            matC[i * MATRIX_SIZE + j] = 0.0f;
        }

        // ѭ����������B����
        for (int j = 0; j < MATRIX_SIZE; ++j)
        {
            // Ϊ����A�ĵ�ǰԪ�ش���һ��512λ����
            __m512 vecA = _mm512_set1_ps(matA[i][j]);

            // ѭ����������B���У����������Ϊ����
            for (int k = 0; k < MATRIX_SIZE; k += VECTOR_WIDTH)
            {
                // ���ڴ��м��ؾ���B������
                __m512 vecB = _mm512_loadu_ps(&matB[j][k]);
                // ���ڴ��м��ؽ�����������
                __m512 vecC = _mm512_loadu_ps(&matC[i * MATRIX_SIZE + k]);
                // ʹ��FMAָ��ִ�о���˷��ۼӲ���
                vecC = _mm512_fmadd_ps(vecA, vecB, vecC);
                // ���������ڴ�
                _mm512_storeu_ps(&matC[i * MATRIX_SIZE + k], vecC);
            }
        }
    }
    return NULL;
}

// ��ʼ������
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

// ��ӡ����
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

// ������
int main()
{
    pthread_t threads[NUM_THREAD];
    ThreadPool threadPool;
    clock_t start, end;
    int prot = PROT_READ | PROT_WRITE;
    int flags = MAP_SHARED | MAP_ANONYMOUS;
    int fd = -1;
    off_t offset = 0;

    // ��ʼ������
    createMatrix();
    // ��ʼ���̳߳�
    initializeThreadPool(&threadPool, NUM_THREAD);

    // ���������ڴ�ӳ���������ñ�������
    matC = mmap(NULL, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, prot, flags, fd, offset);
    if (matC == MAP_FAILED)
    {
        perror("mmap");
        exit(1);
    }

    // ��¼��ʼʱ��
    start = clock();

    // �����̲߳�ִ��AVX-512����˷�
    for (int i = 0; i < NUM_THREAD; ++i)
    {
        pthread_create(&threads[i], NULL, multiplicationAVX512, (void *)&threadPool);
    }

    // �ȴ��߳̽���
    for (int i = 0; i < NUM_THREAD; ++i)
    {
        pthread_join(threads[i], NULL);
    }

    // ��¼����ʱ��
    end = clock();

    printf("\n\nʹ�õ��߳��� -> %d\n", NUM_THREAD);
    printf("\n�����С -> %d\n", MATRIX_SIZE);
    printf("\n��������ʱ��(����) -> %f\n\n", (float)(end - start) * 1000 / CLOCKS_PER_SEC);
    // �ͷŹ����ڴ�ӳ������
    munmap(matC, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    return 0;
}

