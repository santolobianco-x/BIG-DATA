from pyspark import RDD

def matrix_vector_multiply(matrix_by_row: RDD, vector_broadcast) -> RDD:

    def multiply_row(row_data):
        row, entries = row_data

        result = sum(
            value * vector_broadcast.value[col]
            for col, value in entries
        )

        return row, result
    
    return matrix_by_row.map(multiply_row)
