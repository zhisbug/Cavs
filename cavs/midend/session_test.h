#ifndef CAVS_MIDEND_SESSION_TEST_H_
#define CAVS_MIDEND_SESSION_TEST_H_

#include "cavs/midend/session.h"

namespace cavs{

namespace test{

class SessionTest : public SessionBase {
 public:
  SessionTest() {}
  void Run() override {}
};

} //namespace test

} //namespace cavs
#endif
