Statistical Process Control with Python
========================
This Python package implements various methods from the field of Statistical Process Control.

Most of the work is based on the book "Statistical Quality Control" from 2013, 7th Edition, 
by Douglas C. Montgomery. All references to equations and otherwise are to this book if
not stated otherwise.

It is a work in progress and as such, there may be procedures from the field that are yet
to be implemented. The aim is to implement the most typical and useful procedures covered 
by the book. In some cases, where appropriate, useful extensions of these with e.g. PCA 
based preprocessing is also implemented.

The aim of the procedures is to have a consistent and standardized, easy-to-use API, so
that each procedure is used in the same way and with the same calls, attributes etc. Some 
inspiration for this comes from the popular scikit-learn library, where most procedures have
a fit() method that does the heavy lifting by estimating necessary model parameters. This
is the case for this package as well.
