#ifndef _MSC_VER
#ifndef SEMAPHORESET_H
#define SEMAPHORESET_H
#include <string>
#include <ctime>

//namespace Teratomo_SDK
//{

class SemaphoreSet {
public:
	SemaphoreSet(std::string path, int semaphoreNum);
	virtual ~SemaphoreSet();
public:
	virtual int lock();
	virtual int lockSpecific(const int index);
	virtual void unlock();
	void print();
protected:
	bool lock(int semaphore);
private:
	int m_sid; 
	int m_semnum;
	int m_locknum;
};

class SemaphoreSet_with_Timing : public SemaphoreSet {
public:
	SemaphoreSet_with_Timing(std::string path, int semaphoreNum);
	virtual int lock();
	virtual int lockSpecific(const int index);
	virtual void unlock();
	double lockedTime();
private:
	clock_t m_start;
	clock_t m_finish;
};
//}

#endif //SEMAPHORESET_H
#endif
