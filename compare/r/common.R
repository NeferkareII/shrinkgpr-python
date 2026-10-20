# Shared helpers for the R side of the R/Python comparison.
# Tensors travel as JSON objects {"__tensor__": true, "shape": [...], "data": [...]}
# with data flattened in row-major order.

lib <- Sys.getenv("SHRINKGPR_R_LIB")
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({
  library(jsonlite)
  library(torch)
  library(shrinkGPR)
})

read_input <- function(path) fromJSON(path, simplifyVector = FALSE)

write_output <- function(x, path) {
  writeLines(toJSON(x, auto_unbox = TRUE, digits = NA, na = "null", null = "null"), path)
}

is_tensor_json <- function(x) is.list(x) && isTRUE(x[["__tensor__"]])

dec <- function(x, dtype) {
  if (is.null(x)) return(NULL)
  v <- as.numeric(unlist(x$data))
  shp <- as.integer(unlist(x$shape))
  if (length(shp) == 0) return(torch_scalar_tensor(v, dtype = dtype))
  torch_tensor(v, dtype = dtype)$reshape(shp)
}

enc <- function(x) {
  if (is.null(x)) return(NULL)
  if (inherits(x, "torch_tensor")) {
    v <- as.numeric(as_array(x$detach()$cpu()$to(dtype = torch_float64())$flatten()))
    return(list(`__tensor__` = TRUE, shape = as.list(as.integer(x$shape)), data = I(v)))
  }
  if (is.numeric(x)) {
    shp <- if (!is.null(dim(x))) dim(x) else if (length(x) == 1) integer(0) else length(x)
    return(list(`__tensor__` = TRUE, shape = as.list(as.integer(shp)), data = I(as.numeric(x))))
  }
  if (is.list(x)) return(lapply(x, enc))
  x
}

# R's robust_hyperu converts the gsl result with torch_tensor(), which always
# creates float32, so the in-place assignment fails for float64 inputs. For the
# float64 comparison only, make that conversion keep the input dtype.
patch_hyperu_dtype <- function() {
  ns <- asNamespace("shrinkGPR")
  src <- deparse(get("robust_hyperu", ns))
  src <- gsub("device = device)", "device = device, dtype = z$dtype)", src, fixed = TRUE)
  patched <- eval(parse(text = src))
  environment(patched) <- ns
  unlockBinding("robust_hyperu", ns)
  assign("robust_hyperu", patched, ns)
  lockBinding("robust_hyperu", ns)
}

set_precision <- function(name) {
  if (name == "float64") {
    torch_set_default_dtype(torch_float64())
    patch_hyperu_dtype()
    torch_float64()
  } else {
    torch_float32()
  }
}

# Names that only exist in the Python port and have no R counterpart
python_only <- function(name) grepl("(^|\\.)(positive|perm)$", name)

load_state <- function(module, state, dtype) {
  params <- module$named_parameters()
  bufs <- module$named_buffers()
  with_no_grad({
    for (nm in names(state)) {
      t <- dec(state[[nm]], dtype)
      if (!is.null(params[[nm]])) {
        params[[nm]]$copy_(t)
      } else if (!is.null(bufs[[nm]])) {
        bufs[[nm]]$copy_(t)
      } else if (!python_only(nm)) {
        stop("Unknown state entry: ", nm)
      }
    }
  })
}

grads_of <- function(module) {
  g <- lapply(module$named_parameters(), function(p) p$grad)
  names(g) <- paste0("grad.", names(g))
  g
}

state_of <- function(module) {
  s <- module$named_parameters()
  names(s) <- paste0("state.", names(s))
  s
}

kernels <- list(kernel_se = kernel_se, kernel_matern_12 = kernel_matern_12,
                kernel_matern_32 = kernel_matern_32, kernel_matern_52 = kernel_matern_52)
flows <- list(sylvester = sylvester, iaf = iaf, nsf_cl = nsf_cl)
