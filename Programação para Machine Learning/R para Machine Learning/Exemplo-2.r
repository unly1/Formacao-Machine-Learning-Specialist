primeiro.vetor1 <- c(1, 3, 5, 7, 9, 10)
primeiro.vetor2 <- c(1, 4, 5, 8, 22)

# Junta os dois vetores em um terceiro
primeiro.vetor3 <- c(primeiro.vetor1, primeiro.vetor2)

# Calcula e mostra a média do terceiro vetor
media <- mean(primeiro.vetor3)
print(media)

# Exportando para arquivo .txt
writeLines(as.character(media), "Exemplo-2.txt")
