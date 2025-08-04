#include "stdafx.h"
#include "ThreadPool.h"


CThreadPool::CThreadPool(size_t nThreadCnt)
	: m_bStop(false)
{
	m_workers.reserve(nThreadCnt);
	for (size_t i = 0; i < nThreadCnt; i++)
	{
		m_workers.emplace_back(&CThreadPool::run, this);
		SetThreadPriority(m_workers.at(i).native_handle(), THREAD_PRIORITY_LOWEST);
	}
}

CThreadPool::~CThreadPool()
{
	{
		std::lock_guard<std::mutex> guard(m_mtx);

		m_bStop = true;
	}

	m_cv.notify_all();
	for (auto& worker : m_workers)
	{
		if (worker.joinable())
			worker.join();
	}
}

void CThreadPool::ProcessTask(CTask task)
{
	{
		std::lock_guard<std::mutex> guard(m_mtx);

		m_tasks.push(std::move(task));
	}

	m_cv.notify_one();
}

void CThreadPool::run()
{
	bool bStop = false;

	while (!bStop)
	{
		std::unique_lock<std::mutex> lock(m_mtx);
		auto timeout = std::chrono::steady_clock::now();
		bool bTaskReady = false;

		while (!bTaskReady)
		{
			bTaskReady = m_cv.wait_until(lock, timeout, [this, &timeout, &bStop] {
				bStop = m_bStop;
				if (bStop)
					return true;
				if (!m_tasks.empty())
				{
					timeout = m_tasks.top().GetTimePoint();
					return timeout <= std::chrono::steady_clock::now();
				}
				timeout = std::chrono::steady_clock::now() + std::chrono::hours(1);

				return false;
				});
			if (bStop)
				return;
		}

		auto task = m_tasks.top();
		m_tasks.pop();

		lock.unlock();
		task();
	}
}