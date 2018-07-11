/*
 * 
 * Ripser: a lean C++ code for computation of Vietoris-Rips persistence barcodes
 * 
 * Copyright 2015-2016 Ulrich Bauer.
 * 
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program. If not, see
 * <http://www.gnu.org/licenses/>.
 */

/* #define MPD */
/* #define ASSEMBLE_REDUCTION_MATRIX */
#define PRINT_PERSISTENCE_PAIRS

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <unordered_map>

template <class Key, class T> class hash_map : public std::unordered_map<Key, T> {};

/* 
 * So I guess he went through and re-defined a bunch of types so he
 * could change them easily. Maybe that's a good thing in this case:
 * I've never been a fan of re-defining builtin types: I find it
 * obfuscating
 */
typedef float value_t;
typedef int64_t index_t;
typedef int16_t coefficient_t;

/* 
 * This implements a simple table lookup for binomial coefficients. 
 */
class binomial_coeff_table {
  std::vector<std::vector<index_t>> B;
  index_t n_max, k_max;

public:
  binomial_coeff_table(index_t n, index_t k) {
    n_max = n;
    k_max = k;

    B.resize(n + 1);
    for (index_t i = 0; i <= n; i++) {
      B[i].resize(k + 1);
      for (index_t j = 0; j <= std::min(i, k); j++) {
	if (j == 0 || j == i)
	  B[i][j] = 1;
	else
	  B[i][j] = B[i - 1][j - 1] + B[i - 1][j];
      }
    }
  }

  index_t operator()(index_t n, index_t k) const {
    assert(n <= n_max);
    assert(k <= k_max);
    return B[n][k];
  }
};

/*
 * This simply makees a table of multiplicative inverses in a finite
 * field which, for a F2, is 1-x
 */
std::vector<coefficient_t> multiplicative_inverse_vector(const coefficient_t m) {
  std::vector<coefficient_t> inverse(m);
  inverse[1] = 1;

  /* The math:
   *  m = a * (m / a) + m % a
   *
   * so that multipying with inverse(a) * inverse(m % a):
   *
   *  0 = inverse(m % a) * (m / a)  + inverse(a)  (mod m) 
   */
  for (coefficient_t a = 2; a < m; ++a)
    inverse[a] = m - (inverse[m % a] * (m / a)) % m;
  return inverse;
}

/* 
 *
 * What this does is find the value of v in v choose k such that v
 * choose k is less than or equal to idx and v+1 choose k is greater
 * than idx. It presumes that on input v choose k is >= idx on input.
 */
index_t get_next_vertex(index_t& v,
			const index_t idx,
			const index_t k,
			const binomial_coeff_table& binomial_coeff) {

  /* 
   * Does this do a binary search looking for the value of n in
   * nchoosek that such that idx is between nchoosek and nchoose(k+1)? 
   * Is that what it does?
   */
  if (binomial_coeff(v, k) > idx) {
    index_t count = v;
    while (count > 0) {
      index_t i = v;
      index_t step = count >> 1;
      i -= step;
      if (binomial_coeff(i, k) > idx) {
	v = --i;
	count -= step + 1;
      } else {
	count = step;
      }
    }
  }

  assert(binomial_coeff(v, k) <= idx);
  assert(binomial_coeff(v + 1, k) > idx);
  return v;
}

/* 
 * Apparently this gets all the vertices in a simplex 
 */
template <typename OutputIterator>
OutputIterator get_simplex_vertices(index_t idx,
				    const index_t dim,
				    index_t v,
                                    const binomial_coeff_table& binomial_coeff,
				    OutputIterator out) {
  --v;
  for (index_t k = dim + 1; k > 0; --k) {
    get_next_vertex(v, idx, k, binomial_coeff);
    *out++ = v;
    idx -= binomial_coeff(v, k);
  }
  return out;
}

/* 
 * And this function loads a vector with those vertices using the
 * above function 
*/
std::vector<index_t> vertices_of_simplex(const index_t simplex_index,
					 const index_t dim,
					 const index_t n,
                                         const binomial_coeff_table& binomial_coeff) {
  std::vector<index_t> vertices;
  get_simplex_vertices(simplex_index,
		       dim,
		       n,
		       binomial_coeff,
		       std::back_inserter(vertices));
  return vertices;
}

/* 
 * OK, there are multiple definitions depending on wether we are using
 * coefficients or not. This is pretty important I think so I am going
 * to comment this pretty exhaustively, but only in the
 * non-USE_COEFFICIENTS case since at this time I cannot see what the
 * purpose of using them would be
 *
 * An entry is an index type. Why? I have no idea
*/
typedef index_t entry_t;

/*
 * So these functions are overloaded so many times it's hard to
 * reverse-engineer what's going on here. So I attempt to accumulate
 * them here to figure out what they do
 *
 * get_diameter(diameter_index_t) returns a value_t
 * get_index(entry_t) returns an index_t that is the argument
 * get_index(diameter_index_t) returns an index_t that is the first of the pair
 * get_index(diameter_entry_t) returns an index_t
 * get_entry(diamter_entry_t) returns an entry_t&
 * get_entry(entry_t&) returns an entry_t that is the argument
 * get_coefficient(entry_t) returns an index_t that is it's argument
 */
const index_t get_index(entry_t i) { return i; }
index_t get_coefficient(entry_t i) { return 1; }

/* This simple "casts" the index to an entry */
entry_t make_entry(index_t _index, coefficient_t _value) {
  return entry_t(_index);
}

/* This does nothing at all */
void set_coefficient(index_t& e, const coefficient_t c) {}

/*
 * The first defintion of get_entry, for a const entry_t input, that
 * returns that input.
 */
const entry_t& get_entry(const entry_t& e) { return e; }

template <typename Entry> struct smaller_index {
  bool operator()(const Entry& a, const Entry& b) {
    return get_index(a) < get_index(b); }
};

/* 
 * OK, this is actually an entry into a vector that has the diameter
 * and the index of that diameter. I don't know if this is the
 * diameter for a filtration or the diamter between two points
 */
class diameter_index_t : public std::pair<value_t, index_t> {
public:
  diameter_index_t() : std::pair<value_t, index_t>() {}
  diameter_index_t(std::pair<value_t, index_t> p) : std::pair<value_t, index_t>(p) {}
};

/* 
 * This defines the meaning of the first and second entry in the
 * generic pair. This is the first of many get_index methods, this one
 * for a diamter index entry
 */
value_t get_diameter(diameter_index_t i) { return i.first; }
index_t get_index(diameter_index_t i) { return i.second; }

/* 
 * I take it this is an entry into a diameter vector? The constructors support:
 *
 * 1) initializion from a pair
 * 2) initialiation of just the entry if supplied with an entry_type
 * 3) initialization with a diameter value, index value, and coefficent. 
 *    (What is that coefficient? Is that binary coefficient?)
 * 4) Initialization with a diameter index and a coefficient
 * 5) just a diameter index, which uses a coefficient of 1
 */
class diameter_entry_t : public std::pair<value_t, entry_t> {
public:
  diameter_entry_t(std::pair<value_t, entry_t> p) : std::pair<value_t, entry_t>(p) {}
  diameter_entry_t(entry_t e) : std::pair<value_t, entry_t>(0, e) {}
  diameter_entry_t() : diameter_entry_t(0) {}
  diameter_entry_t(value_t _diameter, index_t _index, coefficient_t _coefficient)
    : std::pair<value_t, entry_t>(_diameter, make_entry(_index, _coefficient)) {}
  diameter_entry_t(diameter_index_t _diameter_index, coefficient_t _coefficient)
    : std::pair<value_t, entry_t>(get_diameter(_diameter_index),
				  make_entry(get_index(_diameter_index), _coefficient)) {}
  diameter_entry_t(diameter_index_t _diameter_index) : diameter_entry_t(_diameter_index, 1) {}
};

/* 
 * A get entry method for a diameter entry reference, which returns
 * the second part of the pair: one for const and not to retain const-ness. 
 */
const entry_t& get_entry(const diameter_entry_t& p) { return p.second; }
entry_t& get_entry(diameter_entry_t& p) { return p.second; }

/* 
 * A get index method for diameter entries (not to be confused with
 * diameter index above) that returns the
 */
const index_t get_index(const diameter_entry_t& p) {
  return get_index(get_entry(p));
}
const coefficient_t get_coefficient(const diameter_entry_t& p) {
  return get_coefficient(get_entry(p));
}

const value_t& get_diameter(const diameter_entry_t& p) {
  return p.first;
}

void set_coefficient(diameter_entry_t& p, const coefficient_t c) {
  set_coefficient(get_entry(p), c);
}

/*
 * So this is a structure template that has a single operator defined
 * for it that returns true if the diamiter of a is smaller than b or
 * if they have the same diameter whichever one has the smaller index. 
 */
template <typename Entry> struct greater_diameter_or_smaller_index {
  bool operator()(const Entry& a, const Entry& b) {
    return (get_diameter(a) > get_diameter(b)) ||
      ((get_diameter(a) == get_diameter(b)) && (get_index(a) < get_index(b)));
  }
};

/* 
 * This class template does a lot of the actual grunt work of
 * determining what is close to what. The input DistanceMatrix is a
 * class of distance matrix, in this implementation near as I can tell
 * it is always a compressed_distance_matrix so why we don't just use
 * that is a question I have.
 */
template <typename DistanceMatrix> class rips_filtration_comparator {

public:
  const DistanceMatrix& dist;
  const index_t dim;

private:
  mutable std::vector<index_t> vertices;
  const binomial_coeff_table& binomial_coeff;

public:

  /* 
   * The constructor just initializes the members. What I dont
   * understand now is the initializatino of the vertices vector to
   * the dimension + 1. Doesnt' that leave it full of zeros or
   * something?
   */
  rips_filtration_comparator(const DistanceMatrix& _dist,
			     const index_t _dim,
			     const binomial_coeff_table& _binomial_coeff)  :
    dist(_dist),
    dim(_dim),
    vertices(_dim + 1),
    binomial_coeff(_binomial_coeff){};

  value_t diameter(const index_t index) const {
    value_t diam = 0;
    get_simplex_vertices(index,
			 dim,
			 dist.size(),
			 binomial_coeff,
			 vertices.begin());

    for (index_t i = 0; i <= dim; ++i) {
      for (index_t j = 0; j < i; ++j) {
	diam = std::max(diam, dist(vertices[i], vertices[j]));
      }
    }
    
    return diam;
  }

  /* 
   * This overloads the boolean operator with the template for the
   * diameter/index comparator function
   */
  bool operator()(const index_t a, const index_t b) const {
    assert(a < binomial_coeff(dist.size(), dim + 1));
    assert(b < binomial_coeff(dist.size(), dim + 1));

    return greater_diameter_or_smaller_index<diameter_index_t>()(diameter_index_t(diameter(a), a),
								 diameter_index_t(diameter(b), b));
  }

  template <typename Entry> bool operator()(const Entry& a, const Entry& b) const {
    return operator()(get_index(a), get_index(b));
  }
};

/* This class is where it spends a lot of it's time */
template <class DistanceMatrix> class simplex_coboundary_enumerator {
private:
  index_t idx_below, idx_above, v, k;
  std::vector<index_t> vertices;
  const diameter_entry_t simplex;
  const coefficient_t modulus;
  const DistanceMatrix& dist;
  const binomial_coeff_table& binomial_coeff;

public:
  simplex_coboundary_enumerator(const diameter_entry_t _simplex,
				index_t _dim,
				index_t _n,
				const coefficient_t _modulus,
				const DistanceMatrix& _dist,
				const binomial_coeff_table& _binomial_coeff)
    : simplex(_simplex),
      idx_below(get_index(_simplex)),
      idx_above(0),
      v(_n - 1),
      k(_dim + 1),
      modulus(_modulus),
      binomial_coeff(_binomial_coeff), dist(_dist), vertices(_dim + 1) {

    get_simplex_vertices(get_index(_simplex),
			 _dim,
			 _n,
			 binomial_coeff,
			 vertices.begin());
  }

  bool has_next() {
    while ((v != -1) && (binomial_coeff(v, k) <= idx_below)) {
      idx_below -= binomial_coeff(v, k);
      idx_above += binomial_coeff(v, k + 1);

      --v;
      --k;
      assert(k != -1);
    }
    return v != -1;
  }

  index_t next_index() { return idx_above + binomial_coeff(v--, k + 1) + idx_below; }

  diameter_entry_t next() {
    value_t coface_diameter = get_diameter(simplex);
    for (index_t w : vertices) coface_diameter = std::max(coface_diameter, dist(v, w));
    coefficient_t coface_coefficient = (k & 1 ? -1 + modulus : 1) * get_coefficient(simplex) % modulus;
    return diameter_entry_t(coface_diameter, idx_above + binomial_coeff(v--, k + 1) + idx_below,
			    coface_coefficient);
  }
};

enum compressed_matrix_layout { LOWER_TRIANGULAR, UPPER_TRIANGULAR };

/* 
 * This is a template for a class, a compressed distance matrix. Mind
 * you it doesnt' define how it is layed out in memory: that is done
 * with the templates for the init_rows functions below (how
 * obfuscating is that?).
 */
template <compressed_matrix_layout Layout> class compressed_distance_matrix {
public:

  /* 
   * These are the two values: one is the vector of ALL the distances,
   * the other references to the starting element for each row. This
   * is in row major order.
   */
  std::vector<value_t> distances;
  std::vector<value_t*> rows;

  void init_rows();

  /* 
   * This is a construcor when given a vector of distances. It sets
   * the size from the length of the input vecotr, which is (I think)
   * either lower or upper diagonal but we dont' care which becuase
   * the init_rows call, defined below for both possibiltiies of that,
   * will load properly we think.
   */
  compressed_distance_matrix(std::vector<value_t>&& _distances)
    : distances(_distances), rows((1 + std::sqrt(1 + 8 * distances.size())) / 2) {
    assert(distances.size() == size() * (size() - 1) / 2);
    init_rows();
  }

  /* 
   * This is a constructor when initialized with a Distance Matrix,
   * which is I think fully populated unlike above, when it is just
   * lower (upper?) diagonal.
   */
  template <typename DistanceMatrix>
  compressed_distance_matrix(const DistanceMatrix& mat)
    : distances(mat.size() * (mat.size() - 1) / 2), rows(mat.size()) {
    init_rows();

    for (index_t i = 1; i < size(); ++i)
      for (index_t j = 0; j < i; ++j) rows[i][j] = mat(i, j);
  }

  value_t operator()(const index_t i, const index_t j) const;

  size_t size() const { return rows.size(); }
};

/* 
 * The methods for initializing the row pointers for a lower/upper triangular
 * representation into the compressed distance matrix above
 */
template <> void compressed_distance_matrix<LOWER_TRIANGULAR>::init_rows() {
  value_t* pointer = &distances[0];
  for (index_t i = 1; i < size(); ++i) {
    rows[i] = pointer;
    pointer += i;
  }
}
template <> void compressed_distance_matrix<UPPER_TRIANGULAR>::init_rows() {
  value_t* pointer = &distances[0] - 1;
  for (index_t i = 0; i < size() - 1; ++i) {
    rows[i] = pointer;
    pointer += size() - i - 2;
  }
}

/* And similarly the dereference methods for same */
template <> value_t compressed_distance_matrix<UPPER_TRIANGULAR>::operator()(index_t i, index_t j) const {
  if (i > j) std::swap(i, j);
  return i == j ? 0 : rows[i][j];
}
template <> value_t compressed_distance_matrix<LOWER_TRIANGULAR>::operator()(index_t i, index_t j) const {
  if (i > j) std::swap(i, j);
  return i == j ? 0 : rows[j][i];
}

typedef compressed_distance_matrix<LOWER_TRIANGULAR> compressed_lower_distance_matrix;
typedef compressed_distance_matrix<UPPER_TRIANGULAR> compressed_upper_distance_matrix;

/* Ask Noah what this is all about */
class euclidean_distance_matrix {
public:
  std::vector<std::vector<value_t>> points;

  euclidean_distance_matrix(std::vector<std::vector<value_t>>&& _points) : points(_points) {}

  value_t operator()(const index_t i, const index_t j) const {
    return std::sqrt(std::inner_product(points[i].begin(),
					points[i].end(),
					points[j].begin(),
					value_t(),
					std::plus<value_t>(),
					[](value_t u, value_t v) { return (u - v) * (u - v); }));
  }

  size_t size() const { return points.size(); }
};

/* 
 * This would appear to be a class that lets us find the union of two
 * vectors or something when they are maintained as a binary trees.
 */
class union_find {

  std::vector<index_t> parent;
  std::vector<uint8_t> rank;

public:
  union_find(index_t n) : parent(n), rank(n, 0) {
    for (index_t i = 0; i < n; ++i) parent[i] = i;
  }

  index_t find(index_t x) {
    index_t y = x, z = parent[y];
    while (z != y) {
      y = z;
      z = parent[y];
    }
    y = parent[x];
    while (z != y) {
      parent[x] = z;
      x = y;
      y = parent[x];
    }
    return z;
  }

  void link(index_t x, index_t y) {
    x = find(x);
    y = find(y);
    if (x == y) return;
    if (rank[x] > rank[y])
      parent[y] = x;
    else {
      parent[x] = y;
      if (rank[x] == rank[y]) ++rank[y];
    }
  }
};


template <typename Heap> diameter_entry_t pop_pivot(Heap& column,
						    coefficient_t modulus) {
  if (column.empty()) {
    return diameter_entry_t(-1);
  } else {

    /* Store the top of the heap as the pivot */
    auto pivot = column.top();

    /* 
     * Pop the top off the heap and keep popping until we get an entry
     * with a different pivot
     */
    column.pop();
    while (!column.empty() && get_index(column.top()) == get_index(pivot)) {
      column.pop();
      if (column.empty()) {

	/* Oops we ran out of guys: there is no pivot */
	return diameter_entry_t(-1);
      } else {

	/* Remove ths one too!! */
	pivot = column.top();
	column.pop();
      }
    }
    return pivot;
  }
}

/* 
 * This gets the pviot off the healp, which is the top of the heap or
 * something. The modulus is a passthrough to pop_pivot.
 */
template <typename Heap> diameter_entry_t get_pivot(Heap& column,
						    coefficient_t modulus) {
  diameter_entry_t result = pop_pivot(column, modulus);
  if (get_index(result) != -1) column.push(result);
  return result;
}

/* 
 * OK, this would seem to be an implementation of a sparse matrix deal
 * in what appears to be the obvious way: keeping only as many values
 * in the matrix as there are actual values of the vectors. I guess.
 */
template <typename ValueType> class compressed_sparse_matrix {

  /* The first of these is (I think) the limit on the last entry in the row ? */
  std::vector<size_t> bounds;
  std::vector<ValueType> entries;

public:

  /* I think the size is the number of rows */
  size_t size() const { return bounds.size(); }

  typename std::vector<ValueType>::const_iterator cbegin(size_t index) const {
    assert(index < size());
    return index == 0 ? entries.cbegin() : entries.cbegin() + bounds[index - 1];
  }

  typename std::vector<ValueType>::const_iterator cend(size_t index) const {
    assert(index < size());
    return entries.cbegin() + bounds[index];
  }

  template <typename Iterator> void append_column(Iterator begin, Iterator end) {
    for (Iterator it = begin; it != end; ++it) { entries.push_back(*it); }
    bounds.push_back(entries.size());
  }

  void append_column() { bounds.push_back(entries.size()); }

  void push_back(ValueType e) {
    assert(0 < size());
    entries.push_back(e);
    ++bounds.back();
  }

  void pop_back() {
    assert(0 < size());
    entries.pop_back();
    --bounds.back();
  }

  template <typename Collection> void append_column(const Collection collection) {
    append_column(collection.cbegin(), collection.cend());
  }
};

/* So this pushes an entry onto a column heap. */
template <typename Heap> void push_entry(Heap& column,
					 index_t i,
					 coefficient_t c,
					 value_t diameter) {
  entry_t e = make_entry(i, c);
  column.push(std::make_pair(diameter, e));
}

/*
 * This function makes a new set of columns to reduce. The pivot
 * column index input contains the list of pivot columns. This returns
 * (I think) a vector of columns that have 1 s at the same row as the
 * last column (?). At least it kind of smells that way.
 */
template <typename Comparator>
void assemble_columns_to_reduce(std::vector<diameter_index_t>& columns_to_reduce,
                                hash_map<index_t, index_t>& pivot_column_index,
				const Comparator& comp,
				index_t dim,
                                index_t n,
				value_t threshold,
				const binomial_coeff_table& binomial_coeff) {
  /* 
   * This is interesting: the number of simplices is n choose dim +
   * 2. This means that the input pivot_column_index has this many
   * entries.
   */
  index_t num_simplices = binomial_coeff(n, dim + 2);

  /* Clear the columns! This means we are starting from scratch */
  columns_to_reduce.clear();

#ifdef MPD_MORE
  std::cout << " Assembling from pivots: \n";
  
  for (index_t index = 0; index < num_simplices; ++index) {
    auto temp = pivot_column_index.find(index);
    std::cout << "<" << index << ":" << temp << ">\n";
  }
#endif
  
  /* I think that num_simplices is the number of rows */
  for (index_t index = 0; index < num_simplices; ++index) {


    /* If this is the last of the pivot columns  ... */
    if (pivot_column_index.find(index) == pivot_column_index.end()) {

      /* 
       * Get the dimaeter from the comparator (?) and if it is below
       * threshold, add this to the columns to reduce
       */
      value_t diameter = comp.diameter(index);
      if (diameter <= threshold)
	columns_to_reduce.push_back(std::make_pair(diameter, index));
    }
  }

  /* Sort the columns by diameter/index */
  std::sort(columns_to_reduce.begin(),
	    columns_to_reduce.end(),
	    greater_diameter_or_smaller_index<diameter_index_t>());
}

/* 
 * What I have figured out about this:
 *
 * columns_to_reduce is an input of the columns to reduce, which is a
 * vector of simplex_index and diameters (?)
 *
 * pivot_column_index is an output, or at least it's modified. THis
 * appears to me to be the only output of this function.  
 *
 * dim is the dimension of the work
 * n is .... 
 * threshold is the input, the maximum distance allowed
 * modulus and  multiplicative_inverse define the prime field (F2 in our case)
 * dist is the lower triangular distance matrix
 * comp and comp_prev are comparator functions.
 * binomial_coeff is exactly what it says.
 *
 * So whatever this thing does, it loads pivot_column_index
 */
template <typename DistanceMatrix,
	  typename ComparatorCofaces,
	  typename Comparator>
void compute_pairs(std::vector<diameter_index_t>& columns_to_reduce,
		   hash_map<index_t, index_t>& pivot_column_index,
                   index_t dim,
		   index_t n,
		   value_t threshold,
		   coefficient_t modulus,
                   const std::vector<coefficient_t>& multiplicative_inverse,
		   const DistanceMatrix& dist,
                   const ComparatorCofaces& comp,
		   const Comparator& comp_prev,
                   const binomial_coeff_table& binomial_coeff) {

#ifdef PRINT_PERSISTENCE_PAIRS
  std::cout << "persistence intervals in dim " << dim << ":" << std::endl;
#endif

#ifdef ASSEMBLE_REDUCTION_MATRIX
  compressed_sparse_matrix<diameter_entry_t> reduction_coefficients;
#endif

  std::vector<diameter_entry_t> coface_entries;

  /* Go through all the columns we are reducing */
  for (index_t i = 0; i < columns_to_reduce.size(); ++i) {

    /* Get this column */
    auto column_to_reduce = columns_to_reduce[i];

#ifdef ASSEMBLE_REDUCTION_MATRIX
    std::priority_queue<diameter_entry_t, std::vector<diameter_entry_t>, smaller_index<diameter_entry_t>>
      reduction_column;
#endif

    /* 
     * This instatiates a priority queue to hold the working
     * co-boundary. I should know what that is
     */
    std::priority_queue<diameter_entry_t,
			std::vector<diameter_entry_t>,
			greater_diameter_or_smaller_index<diameter_entry_t>>
                        working_coboundary;

    /* This gets the diameter of this column. What does that mean? */
    value_t diameter = get_diameter(column_to_reduce);
    index_t j = i;

    /* 
     * start with a dummy pivot entry with coefficient -1 in order to
     * initialize working_coboundary with the coboundary of the
     * simplex with index column_to_reduce 
     */
    diameter_entry_t pivot(0, -1, -1 + modulus);

#ifdef ASSEMBLE_REDUCTION_MATRIX
    /*  initialize reduction_coefficients as identity matrix */
    reduction_coefficients.append_column();
    reduction_coefficients.push_back(diameter_entry_t(column_to_reduce, 1));
#endif

    bool might_be_apparent_pair = (i == j);

    /* 
     * This loop apparently goes through all the colums to reduce
     * until it finds a pivot 
     */
    do {

      /* So thi sis 1 - get_coefficent(pivot) */
      const coefficient_t factor = modulus - get_coefficient(pivot);

#ifdef ASSEMBLE_REDUCTION_MATRIX
      auto coeffs_begin = reduction_coefficients.cbegin(j),
	   coeffs_end = reduction_coefficients.cend(j);
#else
      auto coeffs_begin = &columns_to_reduce[j],
	   coeffs_end = &columns_to_reduce[j] + 1;
#endif

      for (auto it = coeffs_begin; it != coeffs_end; ++it) {


	/* 
	 * OK, this is the simplex. I don't get how this 
	 * is parameterized by diameter myself 
	 */
	diameter_entry_t simplex = *it;

	/* Mind you with the prime field F2, this is a no-op */
	set_coefficient(simplex,
			get_coefficient(simplex) * factor % modulus);

#ifdef ASSEMBLE_REDUCTION_MATRIX
	reduction_column.push(simplex);
#endif

	/* Clear the coface entries */
	coface_entries.clear();

	/* 
	 * Create the cofaces of this simplex (?) remember this has a
	 * vector of vertices in it. 
	 */
	simplex_coboundary_enumerator<decltype(dist)> cofaces(simplex,
							      dim,
							      n,
							      modulus,
							      dist,
							      binomial_coeff);

	/* And while there are cofaces ... */
	while (cofaces.has_next()) {

	  /* ... get the coface ... */ 
	  diameter_entry_t coface = cofaces.next();

	  /* ...and if the diamter of that coface is below threshold ... */
	  if (get_diameter(coface) <= threshold) {

	    /* Add this coface to the coface entries */
	    coface_entries.push_back(coface);

	    /*
	     * OK .. I can't believe anybody who holds a degree in CS
	     * or engineering or CE or math would EVER use a goto!!!
	     * But apparently what this does is terminate the loop
	     * when the coface as the same diameter as the simplex (?)
	     * and is the last one colum index. The use of "might be
	     * apparent pair" limits this check to the i == j case. Or
	     * someting.
	     */
	    if (might_be_apparent_pair &&
		(get_diameter(simplex) == get_diameter(coface))) {
	      if (pivot_column_index.find(get_index(coface)) ==
		  pivot_column_index.end()) {
		pivot = coface;
		goto found_persistence_pair;
	      }

	      /* OK, this apparently makes that not possible */
	      might_be_apparent_pair = false;
	    }
	  }
	}

	/* This puts all the coface entries into the working coboundary */
	for (auto e : coface_entries) working_coboundary.push(e);
      }

      /* 
       * get_pivot pops the pivot off of the heap that is the
       * coboundary and removes any other entries that are the same
       * (?)
       */
      pivot = get_pivot(working_coboundary, modulus);

      /* If no pivot is returned ...*/
      if (get_index(pivot) != -1) {

	/* Get the pair ? and make sure it's not the last one */
	auto pair = pivot_column_index.find(get_index(pivot));

	if (pair != pivot_column_index.end()) {
	  j = pair->second;
	  continue;
	}
      } else {
#ifdef PRINT_PERSISTENCE_PAIRS
	std::cout << " [" << diameter << ", )" << std::endl << std::flush;
#endif
	break;
      }

      
    
    found_persistence_pair:
#ifdef PRINT_PERSISTENCE_PAIRS
      value_t death = get_diameter(pivot);
      if (diameter != death) {
	std::cout << " [" << diameter << "," << death << ")" << std::endl << std::flush;
      }
#endif

      /* Add this pivot to the pivot column index, which is the output */
      pivot_column_index.insert(std::make_pair(get_index(pivot), i));
#ifdef ASSEMBLE_REDUCTION_MATRIX
      /*  replace current column of reduction_coefficients (with a single diagonal 1 entry) */
      /*  by reduction_column (possibly with a different entry on the diagonal) */
      reduction_coefficients.pop_back();
      while (true) {
	diameter_entry_t e = pop_pivot(reduction_column, modulus);
	if (get_index(e) == -1) break;
	reduction_coefficients.push_back(e);
      }
#endif
      break;
    } while (true);
  }

}

enum file_format { LOWER_DISTANCE_MATRIX,
		   UPPER_DISTANCE_MATRIX,
		   DISTANCE_MATRIX,
		   POINT_CLOUD,
		   DIPHA };

template <typename T> T read(std::istream& s) {
  T result;
  s.read(reinterpret_cast<char*>(&result), sizeof(T));
  return result; /*  on little endian: boost::endian::little_to_native(result); */
}

compressed_lower_distance_matrix read_point_cloud(std::istream& input_stream) {
  std::vector<std::vector<value_t>> points;

  std::string line;
  value_t value;
  while (std::getline(input_stream, line)) {
    std::vector<value_t> point;
    std::istringstream s(line);
    while (s >> value) {
      point.push_back(value);
      s.ignore();
    }
    if (!point.empty()) points.push_back(point);
    assert(point.size() == points.front().size());
  }

  euclidean_distance_matrix eucl_dist(std::move(points));

  index_t n = eucl_dist.size();

  std::cout << "point cloud with " << n << " points in dimension " << eucl_dist.points.front().size() << std::endl;

  std::vector<value_t> distances;

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j)
      distances.push_back(eucl_dist(i, j));

  return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_lower_distance_matrix(std::istream& input_stream) {
  std::vector<value_t> distances;
  value_t value;
  while (input_stream >> value) {
    distances.push_back(value);
    input_stream.ignore();
  }

  return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_upper_distance_matrix(std::istream& input_stream) {
  std::vector<value_t> distances;
  value_t value;
  while (input_stream >> value) {
    distances.push_back(value);
    input_stream.ignore();
  }

  return compressed_lower_distance_matrix(compressed_upper_distance_matrix(std::move(distances)));
}

compressed_lower_distance_matrix read_distance_matrix(std::istream& input_stream) {
  std::vector<value_t> distances;

  std::string line;
  value_t value;
  for (int i = 0; std::getline(input_stream, line); ++i) {
    std::istringstream s(line);
    for (int j = 0; j < i && s >> value; ++j) {
      distances.push_back(value);
      s.ignore();
    }
  }

  return compressed_lower_distance_matrix(std::move(distances));
}

compressed_lower_distance_matrix read_dipha(std::istream& input_stream) {
  if (read<int64_t>(input_stream) != 8067171840) {
    std::cerr << "input is not a Dipha file (magic number: 8067171840)" << std::endl;
    exit(-1);
  }

  if (read<int64_t>(input_stream) != 7) {
    std::cerr << "input is not a Dipha distance matrix (file type: 7)" << std::endl;
    exit(-1);
  }

  index_t n = read<int64_t>(input_stream);

  std::vector<value_t> distances;

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i > j)
	distances.push_back(read<double>(input_stream));
      else
	read<double>(input_stream);

  return compressed_lower_distance_matrix(std::move(distances));
}

/*
 * Dig it!! They did all that work to overload the init_rows so that
 * it could be easily selected via an enumerated type ... and they do
 * a switch anyways!!!  Doesn't it make more sense to read the file in
 * and then switch on the parsing logic rather than define all these
 * templates? If you're going to switch anyways ...
 */ 
compressed_lower_distance_matrix read_file(std::istream& input_stream,
					   file_format format) {
  switch (format) {
  case LOWER_DISTANCE_MATRIX:
    return read_lower_distance_matrix(input_stream);
  case UPPER_DISTANCE_MATRIX:
    return read_upper_distance_matrix(input_stream);
  case DISTANCE_MATRIX:
    return read_distance_matrix(input_stream);
  case POINT_CLOUD:
    return read_point_cloud(input_stream);
  case DIPHA:
    return read_dipha(input_stream);
  }
}

/* The usage: very useful */
void print_usage_and_exit(int exit_code) {
  std::cerr << "Usage: "
	    << "ripser "
	    << "[options] [filename]" << std::endl
	    << std::endl
	    << "Options:" << std::endl
	    << std::endl
	    << "  --help           print this screen" << std::endl
	    << "  --format         use the specified file format for the input. Options are:" << std::endl
	    << "                     lower-distance (lower triangular distance matrix; default)" << std::endl
	    << "                     upper-distance (upper triangular distance matrix)" << std::endl
	    << "                     distance       (full distance matrix)" << std::endl
	    << "                     point-cloud    (point cloud in Euclidean space)" << std::endl
	    << "                     dipha          (distance matrix in DIPHA file format)" << std::endl
	    << "  --dim <k>        compute persistent homology up to dimension <k>" << std::endl
	    << "  --threshold <t>  compute Rips complexes up to diameter <t>" << std::endl
	    << std::endl;

  exit(exit_code);
}


/* Obviously the main routine ... */
int main(int argc, char** argv) {

  const char* filename = nullptr;

  file_format format = LOWER_DISTANCE_MATRIX;

  index_t dim_max = 1;
  value_t threshold = std::numeric_limits<value_t>::max();

  const coefficient_t modulus = 2;

  /* Parse the command line arguments according to the header */ 
  for (index_t i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help") {
      print_usage_and_exit(0);
    } else if (arg == "--dim") {
      std::string parameter = std::string(argv[++i]);
      size_t next_pos;
      dim_max = std::stol(parameter, &next_pos);
      if (next_pos != parameter.size()) print_usage_and_exit(-1);
    } else if (arg == "--threshold") {
      std::string parameter = std::string(argv[++i]);
      size_t next_pos;
      threshold = std::stof(parameter, &next_pos);
      if (next_pos != parameter.size()) print_usage_and_exit(-1);
    } else if (arg == "--format") {
      std::string parameter = std::string(argv[++i]);
      if (parameter == "lower-distance")
	format = LOWER_DISTANCE_MATRIX;
      else if (parameter == "upper-distance")
	format = UPPER_DISTANCE_MATRIX;
      else if (parameter == "distance")
	format = DISTANCE_MATRIX;
      else if (parameter == "point-cloud")
	format = POINT_CLOUD;
      else if (parameter == "dipha")
	format = DIPHA;
      else
	print_usage_and_exit(-1);
    } else {
      if (filename) { print_usage_and_exit(-1); }
      filename = argv[i];
    }
  }

  /* 
   * This opens the input file as a stream. I am so happy C++ decided
   * to re-do the entire I/O syntax etc. What a farce!!
   */
  std::ifstream file_stream(filename);
  if (filename && file_stream.fail()) {
    std::cerr << "couldn't open file " << filename << std::endl;
    exit(-1);
  }

  /* Read said file from either the input file stream or standard in */
  compressed_lower_distance_matrix dist =
    read_file(filename ? file_stream : std::cin, format);

#ifdef MPD
  {
    index_t irow,icol;

    std::cout << "Input Matrix(" << dist.size() << "): \n\n";
    
    for (irow = 0; irow < dist.size(); irow++) {
      std::cout << "  Row(" << irow << "): ";
      for (icol=0;icol <= irow; icol++)
	std::cout << dist(irow,icol) << " ";
      std::cout << "\n";
    }
    std::cout << "\n\n";
  }
#endif
  
  /* 
   * The distance class has a method for returning the "size" which is
   * the number of points i.e number of rows-1 in the compressed
   * distance matrix
   */
  index_t n = dist.size();
  std::cout << "distance matrix with " << n << " points" << std::endl;

  /* 
   * This gives the range of the distances. I like the standard
   * library minmax_element function but what is it with
   * distances.begin() and distances.end()
   */
  auto value_range =
    std::minmax_element(dist.distances.begin(), dist.distances.end());
  std::cout << "value range: ["
	    << *value_range.first << ","
	    << *value_range.second
	    << "]" << std::endl;
  dim_max = std::min(dim_max, n - 2);

#ifdef MPD
  std::cout << "Dim Max: " << dim_max << std::endl;
#endif

  /* 
   * Creates the binomial coefficient bale, This gets passed by
   * reference most of the time 
   */
  binomial_coeff_table binomial_coeff(n, dim_max + 2);

  /* 
   * The multiplicative inverse is apparently either 0 or 1 as long as
   * we are working with modulo 2 math here, which I think is probably
   * always case unles we are working in the "FINITE FIELD" thing. So
   * let's obfuscate the bitwise or by putting the results in an
   * object!!
   */
  std::vector<coefficient_t> multiplicative_inverse(multiplicative_inverse_vector(modulus));

  /* 
   * OK, the columns to reduce is a vector of indices into the
   * diameter thing. I don't underestand the diameter thing yet.
   */
  std::vector<diameter_index_t> columns_to_reduce;


  /* Not sure why this exists in it's own scope but whatever */
  {

    /* 
     * This is the union class, which I presume is related to the
     * formation of the basis cycle matrix from the boundary matrix
     * and the essential matrix. 
     */
    union_find dset(n);

    /* 
     * A vector of diameter index types. and the comparator we are
     * going to use for something or the other.
     */
    std::vector<diameter_index_t> edges;
    rips_filtration_comparator<decltype(dist)> comp(dist, 1, binomial_coeff);

    /* 
     * This is where the edges are computed using the comparator above
     * (?). Again I find this obfuscating: is there any reason at all
     * that we can't just do a double loop computing distances and
     * pushing them ito the stack?
     */
    for (index_t index = binomial_coeff(n, 2); index-- > 0;) {
      value_t diameter = comp.diameter(index);
      if (diameter <= threshold)
	edges.push_back(std::make_pair(diameter, index));    }
    std::sort(edges.rbegin(),
	      edges.rend(),
	      greater_diameter_or_smaller_index<diameter_index_t>());

#ifdef MPD_MORE
    {
      std::cout << "Edges: " << std::endl << std::endl;
      for (auto e : edges) {
	std::cout << "<" << get_index(e) << "," << get_diameter(e) << "> ";
	if (get_index(e)%8 == 7) std::cout << std::endl;
      }
    }
#endif

#ifdef PRINT_PERSISTENCE_PAIRS
    std::cout << "persistence intervals in dim 0:" << std::endl;
#endif

    std::vector<index_t> vertices_of_edge(2);
    for (auto e : edges) {
      vertices_of_edge.clear();
      get_simplex_vertices(get_index(e), 1, n,
			   binomial_coeff,
			   std::back_inserter(vertices_of_edge));
#ifdef MPD_MORE
      std::cout << "E<" << get_index(e) << "," << get_diameter(e) <<
	"> has " << vertices_of_edge.size() << " vertices: " << std::endl;
      std::cout << "("
		<< vertices_of_edge[0] << ","
		<< vertices_of_edge[1] << ")\n";
#endif
      index_t u = dset.find(vertices_of_edge[0]),
	      v = dset.find(vertices_of_edge[1]);

      if (u != v) {
#ifdef PRINT_PERSISTENCE_PAIRS
	if (get_diameter(e) > 0) std::cout << " [0," << get_diameter(e) << ")" << std::endl;
#endif
	dset.link(u, v);
      } else {
	columns_to_reduce.push_back(e);
      }
    }


    /* THese are the columns sorted in increasing diameter order */
#ifdef MPD_MORE
    {
      int count = 0;
      std::cout << "Columns to reduce: " << std::endl;
      
      for (auto d : columns_to_reduce) {
	std::cout << "<" << d.first << "," << d.second << "> ";
        if ((++count)%8 == 7) std::cout << std::endl;
      }
      std::cout << std::endl;
    }
#endif

    std::reverse(columns_to_reduce.begin(), columns_to_reduce.end());

#ifdef PRINT_PERSISTENCE_PAIRS
    for (index_t i = 0; i < n; ++i)
      if (dset.find(i) == i) std::cout << " [0, )" << std::endl << std::flush;
#endif
  }

  for (index_t dim = 1; dim <= dim_max; ++dim) {

    /* 
     * This declares a comparator (?) in the next dimension and this
     * dimension . My guess is that havingt the one for the previous
     * one supports the co-homology thing?
     */
    rips_filtration_comparator<decltype(dist)> comp(dist,
						    dim + 1,
						    binomial_coeff);
    rips_filtration_comparator<decltype(dist)> comp_prev(dist,
							 dim,
							 binomial_coeff);


    /* 
     * So what this does is create a hash map between column and the
     * row? The "reserve" method puts aside that many keys (?)
     */
    hash_map<index_t, index_t> pivot_column_index;
    pivot_column_index.reserve(columns_to_reduce.size());

    /*
     * 
     */
    compute_pairs(columns_to_reduce,
		  pivot_column_index,
		  dim, n,
		  threshold,
		  modulus,
		  multiplicative_inverse,
		  dist,
		  comp,
		  comp_prev,
		  binomial_coeff);

    if (dim < dim_max) {
      assemble_columns_to_reduce(columns_to_reduce,
				 pivot_column_index,
				 comp,
				 dim,
				 n,
				 threshold,
				 binomial_coeff);
    }
  }
}
