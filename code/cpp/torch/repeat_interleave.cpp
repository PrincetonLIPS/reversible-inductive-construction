#include "repeat_interleave.h"
#include <ATen/Parallel.h>

namespace {
template <typename T> void compute_cpu(T *repeat_ptr, T *cumsum_ptr, T *result_ptr, T size) {
    at::parallel_for(0, size, 1, [&](T i_begin, T i_end) {
        for (T i = i_begin; i < i_end; i++) {
            T end = cumsum_ptr[i];
            T size = repeat_ptr[i];
            T start = end - size;
            for (T j = start; j < end; j++) {
                result_ptr[j] = i;
            }
        }
    });
}

void compute_cpu_scope(const int64_t *scope_ptr, int64_t *result_ptr, int64_t size) {
    at::parallel_for(0, size, 1, [&](int64_t i_begin, int64_t i_end) {
        for (int64_t i = i_begin; i < i_end; i++) {
            int64_t start = scope_ptr[2 * i];
            int64_t size = scope_ptr[2 * i + 1];
            int64_t end = start + size;
            for (int64_t j = start; j < end; j++) {
                result_ptr[j] = i;
            }
        }
    });
}

} // namespace

at::Tensor &induc_gen::repeat_interleave_cpu_out(const at::Tensor &repeats, at::Tensor &out) {
    AT_CHECK(out.is_contiguous(), "Output array must be contiguous.");

    auto repeats_ = repeats.contiguous();
    auto cumsum = repeats.cumsum(0);
    compute_cpu(repeats_.data<int64_t>(), cumsum.data<int64_t>(), out.data<int64_t>(),
                repeats.size(0));
    return out;
}

at::Tensor &induc_gen::repeat_interleave_cpu_out_scope(const at::Tensor &scope, at::Tensor &out) {
    AT_CHECK(out.is_contiguous(), "Output array must be contiguous.");

    auto scope_ = scope.contiguous();
    compute_cpu_scope(scope_.data<int64_t>(), out.data<int64_t>(), scope.size(0));
    return out;
}

at::Tensor &induc_gen::repeat_interleave_out_index_scope(const at::Tensor &scope,
                                                         at::Tensor &out) {
    if (scope.device().is_cuda()) {
        AT_CHECK(out.device().is_cuda(), "Output tensor must be CUDA tensor when scope is.");
        return induc_gen::repeat_interleave_gpu_out_scope(scope, out);
    } else {
        return induc_gen::repeat_interleave_cpu_out_scope(scope, out);
    }
}

at::Tensor &induc_gen::repeat_interleave_out_index(const at::Tensor &repeats, at::Tensor &out) {
    if (repeats.device().is_cuda()) {
        AT_CHECK(out.device().is_cuda(), "Output tensor must be CUDA tensor when repeats is.");
        return induc_gen::repeat_interleave_gpu_out(repeats, out);
    } else {
        return induc_gen::repeat_interleave_cpu_out(repeats, out);
    }
}

static at::Tensor make_repeat_index(at::Tensor repeats, int64_t input_size_dim, int64_t out_length) {
    if (repeats.dim() == 0 || (repeats.dim() == 1 && repeats.size(0) == 1)) {
        repeats = repeats.reshape({1}).expand({input_size_dim});
    } else if (repeats.dim() == 1) {
        AT_CHECK(repeats.size(0) == input_size_dim,
                 "repeats must have the same size as input along dim")
    } else if (repeats.dim() == 2) {
        AT_CHECK(repeats.size(0) == input_size_dim,
                 "scope must have the same size as input along dim.");
        AT_CHECK(repeats.size(1) == 2, "scope must have two columns.");
    } else {
        AT_ERROR("repeats must be 0-dim or 1-dim tensor, or 2-dim tensor representing scopes.");
    }

    at::Tensor index = at::empty({out_length}, repeats.options());

    if (repeats.dim() == 2) {
        induc_gen::repeat_interleave_out_index_scope(repeats, index);
    }
    else {
        induc_gen::repeat_interleave_out_index(repeats, index);
    }

    return index;
}

at::Tensor &induc_gen::repeat_interleave_out(at::Tensor &out, const at::Tensor &self,
                                             const at::Tensor &repeats_or_scope,
                                             c10::optional<int64_t> dim) {
    at::Tensor input = self;
    if (!dim) {
        input = self.flatten();
        dim = 0;
    }

    auto index = make_repeat_index(repeats_or_scope, input.size(dim.value()), out.size(0));
    return at::index_select_out(out, input, dim.value(), index);
}

at::Tensor induc_gen::repeat_interleave_out_shape(const at::Tensor &self,
                                                  const at::Tensor &repeats_or_scope,
                                                  int64_t out_length, c10::optional<int64_t> dim) {
    at::Tensor input = self;
    if (!dim) {
        input = self.flatten();
        dim = 0;
    }

    auto index = make_repeat_index(repeats_or_scope, input.size(dim.value()), out_length);
    return at::index_select(self, dim.value(), index);
}
