#include <iostream>
#include <fstream>
#include <mpi.h>
#include <time.h>
#include <stdlib.h>
#include <cstring>
#include <vector>
#include <queue>
#include <vector>

//实现了宽度顺序旁路 （BFS），以及使用 MPI 技术并联。
//为了演示，更改原始顶点没有意义，因此它是值为 0 的“zahardkozhena”。
//并行化基于并行处理每个级别的想法。
//然后，在“边界”相应顶点的通信器中的所有等级之间有一个分布，即是每个顶点的“主人”，他是在下一级处理特定顶点的人。
//当所有等级都有一个空队列时，就会完成并行处理，通过All_Reduce实现。
//该程序通过串行和并行方法实现计算时间的比较。
//启动时，您需要输入顶点数（大于正在运行的分支数），并指定是否需要保留矩阵和距离向量。


// 串行处理
void serial(int n, int* adjacency_matrix, int save)
{
    std::queue<int> q; // BFS 队列
    std::vector<bool> used(n); // visited数组
    std::vector<int> d(n); // 到顶点距离
    
    used[0] = true;
    q.push(0);

    // 直到队列为空 - 查看顶点
    while (!q.empty())
    {
        // 查看队列中头节点并将其从队列中删除
        int v = q.front();
        q.pop();

        // 查看当前顶点的所有相邻要素
        for (int i = 0; i < n; i++)
        {
            // 如果有边并且之前没有访问过
            // 然后进入队列，标记访问并计算距离
            int to = adjacency_matrix[v * n + i];
            if (to == 1 && !used[i])
            {
                used[i] = true;
                q.push(i);
                d[i] = d[v] + 1;
            }
        }
    }

    if (save == 1)
    {
        std::cout << "Saving distance vector to file \"distance_vector.txt\"..." << std::endl << std::flush;
        std::ofstream path_file("distance_vector.txt");
        path_file << "Distance vector, serial, size = " << n << "\n";
        for (int i = 0; i < n; i++)
        {
            path_file << d[i] << " ";
        }
        path_file << "\n\n";
    }
}







// 寻找高层的结点
int find_owner(int n, int size, int val)
{
    int owner = 0;
    int count = 0;
    int distance = n / size;
    while (count + distance <= val && owner != size - 1)
    {
        owner++;
        count += distance;
    }

    return owner;
}

// 用于校正秩内顶点值的辅助方法
int adjust_vertex(int n, int size, int val)
{
    return val - find_owner(n, size, val) * (n / size);
}

// 并行处理
/*
    n 节点数
    adjacency_matrix 邻接矩阵数据
    rank 当前进程编号
    size 并行进程数(CPU核心数)
    save 是否保存结果
*/
void parallel(int n, int* adjacency_matrix, int rank, int size, int save)
{
    int level = 0; // 当前层级
    bool alive = true; // 激活计算的标志
    std::queue<int> fs, ns; // 边-队列和下一级别的队列
    std::vector<bool> used(n); // 传递顶点的矢量。局部用于轻微优化
    std::vector<int> d(n); // 到顶点的距离
    int* sendcounts = (int*)malloc(sizeof(int) * size); // 每个层次的顶点数
    int* displs = (int*)malloc(sizeof(int) * size); // 每个秩的邻接矩阵中的偏移量

    // 计算邻接矩阵中的顶点数和位移
    int count = n;
    for (int i = 0; i < size - 1; i++)
    {
        sendcounts[i] = (n / size) * n;
        displs[i] = (n - count) * n;
        count -= (n / size);
    }
    sendcounts[size - 1] = count * n;
    displs[size - 1] = (n - count) * n;

    // 每个秩的邻接矩阵分布
    int* adjacency_thread = (int*)malloc(sizeof(int) * n * n);
    MPI_Scatterv(adjacency_matrix, sendcounts, displs, MPI_INT, adjacency_thread, n * n, MPI_INT, 0, MPI_COMM_WORLD);

    // 处理初始化以根（零）秩
    if (rank == 0)
    {
        fs.push(0);
        used[0] = true;
        d[0] = level;
    }

    // 处理一直持续到前一层的队列至少有一个层不为空。
    while (alive)
    {
        level++;
        // 当前层的队列不为空，查看顶点
        while (!fs.empty())
        {
            // 查看边队列中最近的结点并将其从队列中删除
            int v = fs.front();
            fs.pop();
            // 查看当前顶点的所有相邻要素
            for (int i = 0; i < n; i++)
            {
                // 如果有边并且之前没有访问过
                // 然后进入下一级的队列并标记访问
                int to = adjacency_thread[adjust_vertex(n, size, v) * n + i];
                if (to == 1 && !used[i])
                {
                    used[i] = true;
                    ns.push(i);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // 按每个层形成具有下一级队列的数组
        bool* send_q = (bool*)calloc(n, sizeof(bool));
        while (!ns.empty())
        {
            int val = ns.front();
            ns.pop();
            send_q[val] = true;
            d[val] = level;
        }

        // 如果初始是根，那么我们接受来自其他层的队列，否则我们发送
        if (rank == 0)
        {
            // 初始化数组以处理所有列
            bool* recv_q = (bool*)calloc(n, sizeof(bool));
            memcpy(recv_q, send_q, sizeof(bool) * n);

            // 处理所有非根层
            for (int i = 1; i < size; i++)
            {
                MPI_Recv(send_q, n, MPI_C_BOOL, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // приём очереди в массив send_q
                for (int j = 0; j < n; j++)
                {
                    // 如果顶点包含在队列中，则在recv_q数组中设置适当的标志，计算距离
                    if (send_q[j] == true)
                    {
                        recv_q[j] = true;
                        used[j] = true;
                        if (d[j] == 0 && j != 0) d[j] = level;
                    }
                }
            }

            // 将他的顶点序号放入fs
            for (int i = 0; i < n / size; i++) if (recv_q[i]) fs.push(i);
            // 将具有下一级队列的阵列分发到所有其他层
            for (int i = 1; i < size; i++) MPI_Send(recv_q, n, MPI_C_BOOL, i, 0, MPI_COMM_WORLD);
            free(recv_q);

        }
        else
        {
            MPI_Send(send_q, n, MPI_C_BOOL, 0, rank, MPI_COMM_WORLD); // 发送下一个 0 级层的队列
            MPI_Recv(send_q, n, MPI_C_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // 接收公共队列的下一层
            // 进入其该顶点的边队列的当前排名
            if (rank != size - 1)
            {
                for (int i = (n / size) * rank; i < (n / size) * (rank + 1); i++) if (send_q[i]) fs.push(i);
            }
            else for (int i = (n / size) * rank; i < n; i++) if (send_q[i]) fs.push(i);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // 判断是否继续计算
        bool send_alive = fs.empty() ? false : true;
        MPI_Allreduce(&send_alive, &alive, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        free(send_q);
    }

    if (rank == 0)
    {
        if (save == 1)
        {
            std::cout << "Saving distance vector to file \"distance_vector.txt\"..." << std::endl << std::flush;
            std::ofstream path_file("distance_vector.txt", std::ios_base::app);
            path_file << "Distance vector, parallel, size = " << n << "\n";
            for (int i = 0; i < n; i++)
            {
                path_file << d[i] << " ";
            }
            path_file << "\n";
        }
    }

    free(adjacency_thread);
    free(sendcounts);
    free(displs);
}

int main(int argc, char* argv[])
{
    //MPI 初始化
    // srand(time(NULL));
    srand(123);
    MPI_Init(&argc, &argv);
    int rank, size;
    
    // MPI_COMM_WORLD内置的包含所有的通信域
    // 返回当前的进程号（存入rank中）
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // 获取该通信域内的总进程数
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n, save;
    int* adjacency_matrix = NULL;
    double start;
    setlocale(LC_ALL, "russian");

    // 读取输入数据
    if (rank == 0)
    {
        std::cout << "Enter N - adjecency matrix size (int > 0): ";
        std::cin >> n;
        std::cout << "Enter 1 to save further results: matrix and distance vector (could take long time and big size) or 0 to skip: ";
        std::cin >> save;
    }

    std::cout << rank << std::endl;

    // 将输入数据发送到所有处理器
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&save, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n == 0)
    {
        if (rank == 0) std::cout << "N must be bigger than 0" << std::endl;
        MPI_Finalize();
        return 0;
    }

    if (size > n)
    {
        if (rank == 0) std::cout << "Please startup program with param -n less or equal to entered N" << std::endl;
        MPI_Finalize();
        return 0;
    }

    // 根生成邻接矩阵并执行串行处理
    if (rank == 0)
    {
        std::cout << "Generating adjacency matrix..." << std::endl << std::flush;
        adjacency_matrix = (int*)malloc(n * n * sizeof(int));

        for (int i = 0; i < n; i++)
        {
            bool connected = false;

            for (int j = i; j < n; j++)
            {
                if (i == j) adjacency_matrix[i * n + j] = 0;
                else if (j == n - 1 && connected == false)
                {
                    adjacency_matrix[i * n + j] = adjacency_matrix[j * n + i] = 1;
                }
                else
                {
                    int r = rand() % 2;
                    int val = (r == 0) ? 1 : 0;
                    if (val == 1) connected = true;
                    adjacency_matrix[i * n + j] = adjacency_matrix[j * n + i] = val;
                }
            }
        }

        // 将矩阵保存到文件
        if (save == 1)
        {
            std::cout << "Saving generated matrix to file \"adjacency_matrix.txt\"..." << std::endl << std::flush;
            std::ofstream path_file("adjacency_matrix.txt");
            path_file << "Adjacency matrix, size = " << n << "x" << n << "\n";
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    path_file << adjacency_matrix[i * n + j] << " ";
                }
                path_file << "\n";
            }
        }

        std::cout << std::endl << "Serial processing, please wait..." << std::endl << std::flush;
        start = MPI_Wtime();
        serial(n, adjacency_matrix, save);
        std::cout << "Seuqential TIME: " << (MPI_Wtime() - start) << " seconds" << std::endl << std::endl;

        start = MPI_Wtime();
        std::cout << "Parallel processing, please wait..." << std::endl;
    }
    parallel(n, adjacency_matrix, rank, size, save);

    if (rank == 0)
    {
        std::cout << "Parallel TIME: " << (MPI_Wtime() - start) << " seconds" << std::endl;
    }

    free(adjacency_matrix);
    MPI_Finalize();
    return 0;
}
/*
#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<time.h>

int main(int argc, char* argv[])
{
    int myid, numprocs, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);        // starts MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);  // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);      // get number of processes
    MPI_Get_processor_name(processor_name, &namelen);

    if (myid == 0) printf("number of processes: %d\n...", numprocs);
    printf("%s: Hello world from process %d \n", processor_name, myid);

    MPI_Finalize();

    return 0;
}

*/