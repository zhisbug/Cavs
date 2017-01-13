#include "cxx/sym.h"

void load(void** doc_word, int offset) {

}

int main() {
  Sym tpc_word = Sym::Variable();
  Sym doc_tpc  = Sym::Variable();
  Sym doc_word = Sym::Placeholder();

  Sym loss = Sym::Square(doc_word-(doc_tpc*tpc_word));
  Sym step1 = loss.optimizer({doc_tpc}, 20, "projection");
  Sym step2 = loss.optimizer({tpc_word}, 20, "projection");

  Session sess;
  int iters = 100;
  for (int i = 0; i < iters; i++) {
    void* doc_word;
    load(&doc_word, i);
    sess.Run({step1, step2}, {{doc_tpc, doc_word}});
  }

  return 0;
}
