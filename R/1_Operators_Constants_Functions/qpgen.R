quickpick <- function(nos){
  for(i in nos){
    lottonums <- 1:49
    numbers <- sample(lottonums, size = 7, replace = TRUE)
    sorted <- sort(numbers)
    print(sorted)
  }
}

quickpick(nos = 1:10)