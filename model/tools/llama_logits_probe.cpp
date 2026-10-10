// Usage: logits <model.gguf> <tok> [tok...]  -> prints logits for every position, one row per line.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char **argv) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model *m = llama_model_load_from_file(argv[1], mp);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 128;
    llama_context *ctx = llama_init_from_model(m, cp);
    const llama_vocab *vocab = llama_model_get_vocab(m);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n = argc - 2;
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i] = atoi(argv[i + 2]);
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    for (int i = 0; i < n; ++i) {
        const float *l = llama_get_logits_ith(ctx, i);
        for (int v = 0; v < n_vocab; ++v) printf("%.6f%c", l[v], v + 1 == n_vocab ? '\n' : ' ');
    }
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
