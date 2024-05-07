__kernel void partialsum_cols(
	__global int *const g_buf,
	__local int *const l_buf,
	__local int *const r_buf,
	int psum_len,
	int n_rows,
	int n_cols,
	int wg_idx
) {
	// NOTE: only works for 0-offset kernels
	size_t lws = get_local_size(wg_idx);  // work size = n_cols / psum_len
	size_t lid = get_local_id(wg_idx); // local work index
	size_t gj  = get_global_id(0);   // col work index, total: n_cols
	size_t gi  = get_global_id(1);   // row work index, total: n_rows
	size_t idx = gi * n_cols + gj;   // buffer idx

	l_buf[lid] = g_buf[idx];

	barrier(CLK_LOCAL_MEM_FENCE);

	if ((lid & (psum_len - 1)) == 0) {
		for (int i = 1; i < psum_len; i++) {
			l_buf[lid + i] += l_buf[lid + i - 1];
		}
	}

	g_buf[idx] = l_buf[lid];

	barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void prefixsum_cols(
	__global int *const g_buf,
	__local int *const l_buf,
	__local int *const r_buf,
	int psum_len,
	int n_rows,
	int n_cols,
	int wg_idx
) {
	// NOTE: only works for 0-offset kernels
	size_t lws = get_local_size(wg_idx);  // work size = n_cols / psum_len
	size_t lid = get_local_id(wg_idx);    // local work index
	size_t gj  = get_global_id(0);   // col work index, total: n_cols / psum_len = work_size
	size_t gi  = get_global_id(1);   // row work index, total: n_rows
	size_t idx;

   // buffer idx
	if (wg_idx == 0) {
		// agg along each row
		idx = gi * n_cols + (gj + 1) * psum_len - 1;
	} else {
		// agg along each column
		idx = ((gi + 1) * psum_len - 1) * n_cols + gj;
	}

	l_buf[lid] = g_buf[idx];

	barrier(CLK_LOCAL_MEM_FENCE);

	// --- downsweep ---
	// dive down and add sums
	// ---
	// i = 1 => x[0] + x[1], x[2] + x[3] ...apply on every 2nd work item
	// i = 2 => x[1] + x[3], x[5] + x[7] ...apply on every 4th work item
	// i = 3 => x[3] + x[7], x[11] + x[15] ...apply on every 8th work item
	// i = n => x[2^(n-1) * 1 - 1] + x[2^(n-1) * 2 - 1], ...
	// test: i = 5 => x[15] + x[31], x[47] + x[63], ...ok.
	// conditions: 2^i - 1 <= lws
	// ---
	// setting i <- i * 2 each iteration so, 2^i in condition above is simply i
	// i.e. i < lws

	for (int i = 2; i <= lws; i = (i << 1)) {
		if ((lid & (i - 1)) == 0 && (lid + i - 1) <= lws) {
			l_buf[lid + i - 1] += l_buf[lid + (i >> 1) - 1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// set last entry to 0 to setup for upsweep
	int tmp_last;
	if (lid == (lws - 1)) {
		tmp_last = l_buf[lid];
		l_buf[lid] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// --- upsweep ---
	// swim up by swap 2^depth-neighbour and sum
	// ---
	// i = 1 => swap(x[N-1], x[N/2-1]), x[N-1] <- x[N-1] + x[N/2-1]
	// i = 2 => N <- N / 2, repeat
	// ...
	int i_swap, i_end, tmp;
	for (int i = lws; i >= 2; i = (i >> 1)) {
		i_swap = (i >> 1) - 1;
		i_end = i - 1;
		if ((lid & i_end) == 0 && (lid + i_end) < lws) {
			tmp = l_buf[lid + i_swap];
			l_buf[lid + i_swap] = l_buf[lid + i_end];
			l_buf[lid + i_end] += tmp;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// --- cycle back and restore last element ---
	if (lid < lws - 1) {
		r_buf[lid] = l_buf[lid + 1];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid == (lws - 1)) {
		r_buf[lid] = tmp_last;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// write back prefix sum into r-w buffer except for lid = 0
	g_buf[idx] = r_buf[lid];
	
}

__kernel void expandsum_cols(
	__global int *const g_buf,
	__local int *const l_buf,
	__local int *const r_buf,
	int psum_len,
	int n_rows,
	int n_cols,
	int wg_idx
) {
	// NOTE: only works for 0-offset kernels
	size_t lws = get_local_size(wg_idx);  // work size = n_cols / psum_len
	size_t lid = get_local_id(wg_idx);    // local work index
	size_t gj  = get_global_id(0);   // col work index, total: n_cols
	size_t gi  = get_global_id(1);   // row work index, total: n_rows
	size_t idx = gi * n_cols + gj;   // buffer idx

	l_buf[lid] = g_buf[idx];

	barrier(CLK_LOCAL_MEM_FENCE);

	// skip first psum_len - 1 elements
	// then, add every psum_len'th element subsequent psum_len - 1 elements
	if ((lid & (psum_len - 1)) == 0 && lid >= psum_len - 1) {
		for (int i = 0; i < psum_len - 1; i++) {
			l_buf[lid + i] += l_buf[lid - 1];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// write back prefix sum into r-w buffer
	g_buf[idx] = l_buf[lid];
}


// scan fss across largest dimension.
__kernel void computefss(
	__global int    *const f_buf,     // read-only
	__global int    *const o_buf,     // read-only
	__global float  *const fss_buf,   // write-only
	int n_rows,
	int n_cols,
	int w_rows,
	int w_cols,
	int wg_idx
) {
	// NOTE: only works for 0-offset kernels
	size_t lws = get_local_size(wg_idx);  // local work size
	size_t lid = get_local_id(wg_idx);    // local work index
	size_t gj  = get_global_id(0);   // col work index, total: n_cols
	size_t gi  = get_global_id(1);   // row work index, total: n_rows
	size_t idx = gi * n_cols + gj;   // buffer idx
	size_t idx_w = gi * (n_cols - w_cols) + gj;

	// --- perform read into local buffers ---
	if (
		gj < (n_cols - w_cols)    // window must be internal to data bounds
		&& gi < (n_rows - w_rows)
		&& lid < lws              // sanity check
	) {
		// ----------------------------
		// A   B
		// +---+
		// |   |
		// +---+
		// C   D   area = D - B - C + A
		// -----------------------------
		int2 _a = (int2)(
			o_buf[idx],
			f_buf[idx]
		);
		int2 _b = (int2)(
			o_buf[idx + w_cols],
			f_buf[idx + w_cols]
		);
		int2 _c = (int2)(
			o_buf[idx + w_rows * n_cols],
			f_buf[idx + w_rows * n_cols]
		);
		int2 _d = (int2)(
			o_buf[idx + w_cols + w_rows * n_cols],
			f_buf[idx + w_cols + w_rows * n_cols]
		);

		int2 area = (_d - _b - _c + _a);
		int denom = area.s0 + area.s1;

		if (denom > 0) {
			fss_buf[idx_w] = 1.0f - (float)abs(area.s0 - area.s1) / (float)denom;
		}
	}
}
