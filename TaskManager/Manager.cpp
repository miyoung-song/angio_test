#include "Manager.h"

namespace TaskMngr {

	CManager::CManager(size_t nMaxWorkers)
		: m_maxWorkers(nMaxWorkers)
		, m_workerCnt(0)
		, m_bStop(false)
	{
		m_threadPool = std::make_shared<CThreadPool>(nMaxWorkers);
	}

	CManager::~CManager()
	{
		
	}

	std::future<void> CManager::Stop()
	{
		auto task = std::make_shared<std::packaged_task<void()>>([this]
			{
				std::unique_lock<std::mutex> guard(m_mtx);
				bool bIsLast = m_workerCnt == 1;

				while (!bIsLast)
				{
					guard.unlock();
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
					guard.lock();
					bIsLast = m_workerCnt == 1;
				}
			});
		auto future = task->get_future();

		auto functor = [task = std::move(task)]() mutable
		{
			(*task)();
		};
		std::lock_guard<std::mutex> guard(m_mtx);

		m_bStop = true;
		m_tasks.emplace(std::move(functor), std::chrono::steady_clock::now());
		this->ProcessTask();

		return future;
	}

	void CManager::AddTask(std::function<void()> func)
	{
		std::lock_guard<std::mutex> guard(m_mtx);

		if (m_bStop)
			return;

		m_tasks.emplace(std::move(func), std::chrono::steady_clock::now());
		this->ProcessTask();
	}

	void CManager::ProcessTask()
	{
		if (m_tasks.empty() || m_workerCnt == m_maxWorkers)
			return;

		auto task = std::move(m_tasks.front());
		m_tasks.pop();

		m_workerCnt++;
		m_threadPool->ProcessTask(std::move(task));
	}
	// 생성할 쓰레드 갯수 지금은 8개 [5/20/2021 Jeon]
	std::unique_ptr<CManager, std::default_delete<CManager>> g_manager = std::make_unique<CManager>(16);

}