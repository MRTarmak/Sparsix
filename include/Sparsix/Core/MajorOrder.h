#pragma once

namespace sparsix {

/** @brief Major dimension used when ordering COO triplets. */
enum class MajorOrder : bool {
    /** @brief Sort by row, then column. */
    RowOrder,
    /** @brief Sort by column, then row. */
    ColumnOrder
};

} // namespace sparsix
