#pragma once

using TimePoint = std::chrono::steady_clock::time_point;

class CTask
{
public:
	CTask(std::function<void()> func, const TimePoint timePoint)
		: m_func(std::move(func))
		, m_timePoint(timePoint) {}

	const TimePoint& GetTimePoint() const { return m_timePoint; };

	bool operator>(const CTask& other) const
	{
		return m_timePoint > other.m_timePoint;
	}

	void operator()()
	{
		m_func();
	}

private:
	std::function<void()> m_func;
	TimePoint m_timePoint;
};