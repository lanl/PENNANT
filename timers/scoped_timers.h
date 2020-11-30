#ifndef SCOPED_TIMERS_H
#define SCOPED_TIMERS_H

#include <string>
#include <chrono>
#include <map>
#include <vector>
#include <numeric>
#include <cstdio>
#include <hip/hip_runtime.h>

#ifdef USE_TIMERS
#define DEVICE_TIMER(group_name, timer_name, stream) \
  scoped_hip_event_timer timer ## __LINE__((group_name), (timer_name), (stream))
#define HOST_TIMER(group_name, timer_name) \
  scoped_system_timer timer ## __LINE__((group_name), (timer_name))
#define TIMER_REPORT() timer::report()
#else
#define DEVICE_TIMER(group_name, timer_name, stream)
#define HOST_TIMER(group_name, timer_name)
#define TIMER_REPORT()
#endif

class time_tracker {
  using timer_data_t = std::map<std::string, std::vector<float>>;
  using group_data_t = std::map<std::string, timer_data_t>;
 public:
  void submit(const std::string& group_name, const std::string& timer_name, float time){
    time_data[group_name][timer_name].push_back(time);
    
  }
  void report(){
    printf("timer stats:\n");
    float grand_total = 0.;
    for(const auto& group_entry: time_data){
      const auto& group_name = group_entry.first;
      const auto& group_data = group_entry.second;
      float group_total = 0.;
      for(const auto& timer_entry: group_data){
	const auto& timer_name = timer_entry.first;
	const auto& timer_data = timer_entry.second;
	auto total = std::accumulate(timer_data.begin(), timer_data.end(), 0.0f);
	group_total += total;
	printf("%s, %s, %.2f seconds\n", group_name.c_str(), timer_name.c_str(), total);
      }
      grand_total += group_total;
      printf("Total time for %s: %.2f seconds\n", group_name.c_str(), group_total);
    }
    printf("Total time for all timer groups: %.2f seconds\n", grand_total);
  }
 private:
  group_data_t time_data;
};


class timer {
public:
  void submit(const std::string& group_name, const std::string& timer_name, float time){
    tracker.submit(group_name, timer_name, time);
  }
  static void report(){ tracker.report(); }

private:
  static time_tracker tracker;
};


class scoped_timer : public timer { // abstract class
public:
  scoped_timer(const std::string& group_name, const std::string& timer_name)
    : group_name(group_name),
      timer_name(timer_name)
  {}
  
  scoped_timer(const scoped_timer&) = delete;
  scoped_timer& operator=(const scoped_timer&) = delete;
  virtual ~scoped_timer() = 0;
  void submit(float time){
    timer::submit(group_name, timer_name, time);
  }
  
private:
  std::string group_name;
  std::string timer_name;
};

inline scoped_timer::~scoped_timer(){}

class scoped_system_timer : public scoped_timer {
public:
  scoped_system_timer(const std::string& group_name, const std::string& timer_name)
    : scoped_timer(group_name, timer_name),
      start(std::chrono::steady_clock::now())
  {}
  ~scoped_system_timer(){
    auto stop = std::chrono::steady_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<float>>(stop - start).count();
    submit(elapsed_seconds);
  }
private:
  std::chrono::steady_clock::time_point start;
};


class scoped_hip_event_timer : public scoped_timer {
public:
  scoped_hip_event_timer(const std::string& group_name, const std::string& timer_name, hipStream_t stream)
    : scoped_timer(group_name, timer_name),
      stream(stream)
  {
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, stream);
  }
  ~scoped_hip_event_timer(){
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);
    float ms;
    hipEventElapsedTime(&ms, start, stop);
    submit(ms / 1000.0);
    hipEventDestroy(stop);
    hipEventDestroy(start);
  }
private:
  hipStream_t stream;
  hipEvent_t start;
  hipEvent_t stop;
};

#endif // SCOPED_TIMERS_H
