#pragma once

#include "Task.h"

class CThreadPool
{
public:
	explicit CThreadPool(size_t nThreadCnt);
	~CThreadPool();

	void ProcessTask(CTask task);

private:

	void run();

	std::priority_queue<CTask, std::vector<CTask>, std::greater<>> m_tasks;
	std::vector<std::thread> m_workers;
	std::condition_variable m_cv;
	std::mutex m_mtx;

	bool m_bStop;
	
};

