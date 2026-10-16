## Resubmission

This is a resubmission. In this version I have:

* Removed extra spaces in the DESCRIPTION, added () after function names, put package and software names in single quotes, and added references in the form authors (year) <doi:...>.
* Added \value to all listed .Rd files, describing the class and meaning of the output.
* Replaced \dontrun{} with \donttest{}, except for two examples that use parallel fitting with 'doMC', which is not available on all platforms.
* Restored the user's par() settings in plot.ptLasso() with on.exit(), and in the MultiResponseMixed vignette.
* Changed the title to "Pretraining the Lasso".
* Reduced the overall check time: six of the vignettes are now website-only, and some test files are skipped on CRAN.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.
* The possibly misspelled words in DESCRIPTION are author names (Hastie, Tibshirani), "et al.", and technical terms (pretraining, interpretability); all are spelled correctly.
