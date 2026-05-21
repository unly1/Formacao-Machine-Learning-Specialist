# Volume de um tubo
# Seja um tubo com raio de 10 cm, com 1,5 metros de comprimento e com uma
# espessura de 1 cm. Qual o volume deste tubo?

raio <- 10 # Obs: Raio em centímetros
comprimento <- 150 # Obs: Comprimento em centímetros
espessura <- 1 # Obs: Espessura em centímetros
volume <- pi * (raio - espessura)^2 * comprimento # Calcula volume do tubo

# Volume em centímetros cúbicos
 print(paste("O volume do tubo é de ", volume, "centímetros cúbicos."))

# Volume em litros
volume_litros <- volume / 1000
print(paste("O volume do tubo é de ", volume_litros, "litros."))

# Exportando resultados para arquivo .txt
texto_cm3 <- paste("O volume do tubo é de ", volume, "centímetros cúbicos.")
texto_litros <- paste("O volume do tubo é de ", volume_litros, "litros.")
writeLines(c(texto_cm3, texto_litros), "Exemplo-1.txt")
