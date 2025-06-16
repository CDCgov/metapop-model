# Defining beta from per capita contacts

## Notation

Consider populations denoted as $A, B, C, \dots$ with total size denoted
$N_A, N_B, N_C, \dots$. Individuals in a group have $k_A, k_B, k_C, \dots$
number of total contacts per day.
Denote $i$ as the population receiving contact and $j$ as the
population providing contact and $K_{ij}$ as the number of interactions
among individuals in populations $i$ and $j$. By definition, $K_{ij}$ is
symmetric, that is, $K_{ij}$ = $K_{ji}$.

For the sake of example, we will consider three populations $A, B, C$ where
$N_A = 1000, N_B = 100, N_C = 100,$ and $k = k_A = k_B = k_C = 10$.

# Determine the number of contacts among groups

Since $K_{ij}$ must equal $K_{ji}$, defining a matrix $K$ among $z$ populations
requires $z \choose 2$ constraints. We will denote these constraints as $k_{ij}$
such that it represents the number of contacts a person in group $j$ has with
people in group $i$.

For our example we need 3 constraints, so let's define them with respect to the
smaller populations being the 'from' population:

$k_{AB} = 2, k_{CB} = 1,$ and $k_{AC} = 2.$

Now we can calculate the cells of $K$ starting with our constraints where

$$K_{ij} = K_{ji} = k_{ij} N_j:$$

$$\begin{matrix}
  & A & B & C \\
A & - & 200 & 200 \\
B & 200 & - & 100 \\
C & 200 & 100 & - \\
\end{matrix}$$

The remaining values can be calculated recognizing that the total number of
contacts for a population are $k_j N_j$ and these values form the margins of
our table. In our example,
$K_{AA} = k N_A - (K_{AB} + K_{AC}) = 10 \times 1000 - (200 + 200) = 9600$,
with $K_{BB}$ and $K_{CC}$ calculated similarly such that:

$$\begin{matrix}
  & A & B & C \\
A & 9600 & 200 & 200 & 10000 \\
B & 200 & 700 & 100 & 1000 \\
C & 200 & 100 & 700 & 1000 \\
  & 10000 & 1000 & 1000 & 12000
\end{matrix}$$

To make the per capita matrix, which we will denote $M$,
we ensure that the 'to' contacts sum properly
to our $k$ for each population:

$$\begin{matrix}
  & A & B & C \\
A & 9.6 & 2 & 2 &  \\
B & 0.2 & 7 & 1 &  \\
C & 0.2 & 1 & 7 &  \\
  & 10 & 10 & 10 &
\end{matrix}$$

which can be easily interpreted as each person in population $A$ has
9.6 of their 10 contacts with people in population $A$ but only $0.4$ of them
with the other two populations, in this example, equally divided.  As expected,
this matrix is not symmetric.

## New infections generated

Let's start from the example and work to derive a general formula for the
number of new infections generated in a population at a given time, i.e. the
number of new infections in a population $i$ based on the number of infectious
individuals present.

Consider here, for the sake of example, that $I_i = [10, 5, 1]$, what is the
number of infections generated in $A$ from infectious people in $A$
($A \rightarrow A$)?

In the beginning there are
$S_A = N_A - I_A - R_A = 1000 - 10 - 0 = 990$ susceptible. Thus, the
probability of one infected person coming in contact with a susceptible person
is $990 / (1000 - 1)$, the $1$ accoutung for the infected individual who is in
the same population and will not make self-contact nor self-infect. Each infected
person (like everyone else) has $k = 10$ contacts of which 9.6 are with other $A$'s.
Thus, a single infected person in $A$ generates on average infections in other
people in $A$ equal to

$$9.6 * 990 / (1000 - 1) = 9.51$$

for a total number of new infections in $A$ from $A$, given there
are 10 infected individuals, of $95.1$.

In general,

$I_j M_{ij} S_i / (N_i - \delta_{ij})$

is the number of new infections in population $i$ due to infectious people in
population $j$, where $\delta_{ij} = 1$ if $i = j$ and $0$ otherwise (i.e., the
Kronecker delta). $\delta_{ij}$ arises because when $i \neq j$ the infected
individual is not in the group and thus the entire 'to' population would be
considered in the denominator.

To calculate the total number of new infections in $i$, sum over all
infectious populations $j$:

$$\sum_j I_j M_{ij} S_i / (N_i - \delta_{ij}) = S_i \sum_j I_j M_{ij} / (N_i - \delta_{ij}).$$
