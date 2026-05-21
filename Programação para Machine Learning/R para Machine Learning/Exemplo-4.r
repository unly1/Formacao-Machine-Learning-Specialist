# Criação de vetor
x <- c(12, 7, 3, 4.2, 18, 2, 54, -21, 8, -5)

# Cálculo da média do vetor
result.mean <- mean(x)
print(result.mean)

# Cálculo da mediana do vetor
result.median <- median(x)
print(result.median)

# Cálculo do desvio padrão do vetor
result.sd <- sd(x)
print(result.sd)

# Cálculo da variância do vetor
result.var <- var(x)
print(result.var)

# Exportando estatísticas para arquivo .txt
estatisticas <- c(
  paste("Média:", result.mean),
  paste("Mediana:", result.median),
  paste("Desvio Padrão:", result.sd),
  paste("Variância:", result.var)
)
writeLines(estatisticas, "Exemplo-4.txt")