#ifndef _MSC_VER
#include <iostream>
#include "SemaphoreSet.h"
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <stdexcept>
#include <sys/errno.h>
#include <sched.h> //ez nem igazan kell
#include <stdio.h>
#include <unistd.h>

//namespace Teratomo_SDK
//{

union semun
{
    int val;                           // value for SETVAL
    struct semid_ds *buf;              // buffer for IPC_STAT & IPC_SET
    unsigned short int *array;         // array for GETALL & SETALL
    struct seminfo *__buf;             // buffer for IPC_INFO
};

    
SemaphoreSet::SemaphoreSet(std::string path, int semaphoreNum)
    :m_sid(-1)
    ,m_semnum(semaphoreNum)
    ,m_locknum(-1)
{
//	TERATOMO_RT_REPORTS_ASSERT_INVALID_OPERATION(access(path.c_str(), F_OK)==0, "SemaphoreSet given to path is not valid");
    key_t key=ftok(path.c_str(),'s');
    m_sid=semget(key,semaphoreNum,IPC_CREAT|IPC_EXCL|0666);
    if (m_sid == -1)
    {
	if (errno != EEXIST)
	    throw std::logic_error("Failed to create semaphore set");
	std::cout<<"The semaphore already exists. sid="<<m_sid<<errno<<std::endl;
	m_sid=semget(key,0,0666);
    }
    else
    {
        std::cout<<"The semaphore was created. sid="<<m_sid<<std::endl;
        union semun semopts;
	semopts.val=1;
        for(int i=0;i<semaphoreNum;++i)
	    semctl(m_sid,i,SETVAL,semopts);
    }
    print();
}

SemaphoreSet::~SemaphoreSet()
{
    if (m_locknum != -1)
	unlock();
}

bool SemaphoreSet::lock(int semaphore)
{
    struct sembuf sem_lock={ semaphore, -1, IPC_NOWAIT|SEM_UNDO};
    if((semop(m_sid, &sem_lock, 1)) == -1)
    {
	return false;
    }
    std::cout<<"Semaphore resources decremented by one (locked)"<<std::endl;
    print();
    return true;
}

int SemaphoreSet::lock()
{
    if (m_locknum != -1)
		throw std::logic_error( "The lock can be called only once." );
    while(true)
    {
		for(int i=0;i<m_semnum;++i)
        {
    	    if (lock(i))
			{
				m_locknum=i;
				return i;
			}
        }
		usleep(100);
//	sched_yield(); //valahogy szarul sleepelte magat regen, ezert probalkoztam ilyennel is.
    }
}

int SemaphoreSet::lockSpecific(const int index)
{
    if (m_locknum != -1)
		throw std::logic_error( "The lock can be called only once." );
	if (index < 0 || index >= m_semnum)
 		throw std::logic_error( "Invalid index specified" );
    while(true)
    {
		if (lock(index))
		{
			m_locknum=index;
			return index;
        }
		usleep(100);
//	sched_yield(); //valahogy szarul sleepelte magat regen, ezert probalkoztam ilyennel is.
    }
}


void SemaphoreSet::unlock()
{
    if (m_locknum == -1)
        throw std::logic_error( "Unlock was called, but no lock was invoked before." );
    struct sembuf sem_unlock={ m_locknum, 1, IPC_NOWAIT|SEM_UNDO};
    if((semop(m_sid, &sem_unlock, 1)) == -1)
    {
	throw std::logic_error( "Unlock failed." );
    }
    m_locknum=-1;
    std::cout<<"Semaphore resources incremented by one (unlocked)"<<std::endl;
    print();
}

void SemaphoreSet::print()
{
    int semval;
    for(int i=0;i<m_semnum;++i)
    {
	semval = semctl(m_sid, i, GETVAL, 0);
        std::cout<<i<<"="<<semval<<" ";
    }
    std::cout<<std::endl;
}

SemaphoreSet_with_Timing::SemaphoreSet_with_Timing(std::string path, int semaphoreNum) :
    SemaphoreSet(path, semaphoreNum)
    , m_start(0)
    , m_finish(0)
{
}

int SemaphoreSet_with_Timing::lock() {
    m_start=clock();
    return (SemaphoreSet::lock());
}

int SemaphoreSet_with_Timing::lockSpecific(const int index) {
    m_start=clock();
    return (SemaphoreSet::lockSpecific(index));
}

void SemaphoreSet_with_Timing::unlock() {
    m_finish=clock();
    SemaphoreSet::unlock();
}



double SemaphoreSet_with_Timing::lockedTime() {
    return((double(m_finish)-double(m_start))/CLOCKS_PER_SEC);
}
//}

#endif
