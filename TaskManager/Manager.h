#pragma once

#include "stdafx.h"
#include "Task.h"
#include "ThreadPool.h"

namespace TaskMngr {

	class CManager
	{
	public:
		CManager(size_t nMaxWorkers);
		~CManager();

		std::future<void> Stop();

		template <class F, class... Args>
		auto push(F&& func, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
		{
			auto task = std::make_shared<std::packaged_task<typename std::result_of<F(Args...)>::type()>>
				(std::bind(std::forward<F>(func), std::forward<Args>(args)...));

			auto future = task->get_future();

			auto functor = [this, task = std::move(task)]() mutable
			{
				(*task)();

				std::lock_guard<std::mutex> guard(m_mtx);

				m_workerCnt--;
				this->ProcessTask();

			};
			this->AddTask(std::move(functor));

			return future;
		};

	private:
		void AddTask(std::function<void()> func);
		void ProcessTask();

		std::shared_ptr<CThreadPool> m_threadPool;
		std::queue<CTask> m_tasks;
		std::mutex m_mtx;
		size_t m_maxWorkers;
		size_t m_workerCnt;

		bool m_bStop;

	};

	extern std::unique_ptr<CManager, std::default_delete<CManager>> g_manager;

}