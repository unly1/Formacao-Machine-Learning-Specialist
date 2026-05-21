A <- matrix(c(2, 4, 3, 1, 5, 7), nrow = 2, ncol = 3, byrow = TRUE)
B <- matrix(c(2, 5, 3, 4, 5, 12), nrow = 2, ncol = 3, byrow = TRUE)

# Multiplicação de matrizes
resultado <- A * B
print(resultado)

# Exportando para arquivo .txt
write.table(resultado, "Exemplo-3.txt", row.names = FALSE, col.names = FALSE)
