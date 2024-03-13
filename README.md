# Retrieval Augmented Autoformalization with Refinement (Auto-correction)

## How to set up IsarMathLib in Isabelle
First, you should download [Isabelle](https://isabelle.in.tum.de/) and add its "bin" to your PATH variable. In Linuxs systems, run
```
export PATH=$Isabelle/bin:$PATH
```
where "$Isabelle" is your Isabelle directory path.

Next, copy IsarMathLib under this repository to Isabelle by running
```
cp -r IsarMathLib $Isabelle/src/ZF
```

Finally, append "ROOT" file under IsarMathLib to "ROOT" file under ZF by running
```
cd $Isabelle/src/ZF
cat IsarMathLib/ROOT >> ROOT
```

Now you should be able to build an "IsarMathLib" session in which you can use all theories from IsarMathLib.
