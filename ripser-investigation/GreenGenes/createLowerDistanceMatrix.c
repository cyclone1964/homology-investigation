/*C
 * createLowerDistanceMatrix - generate the lower distance matrix
 *
 * Usage: 
 *
 * createLowerDistanceMatrix InputFileName OutputFileName 
 *
 * This simple program creates (get this) a lower distance matrix from
 * the first file storing results in the second. The input file is
 * assumed to conform to my reverse-engineering of the .fasta files I
 * downloaded from the GreenGenes data site. The output file is a
 * lower-diagonal distance matrix.
 *
 * The .fasta file has a series of "header lines" that give the name of the
 * strain and some other ancillary data. I dont' now what that is, but
 * the line has (it appears)
 *
 * 1) a ">" in the first column
 * 2) a number immediately after that which I ignore
 * 3) a series of words after that that represent the name of the species
 *
 * Then it has lines of alignment data: no assumption is made about
 * the number of these or their length other than a maximal
 * length. These are long streams of A, T, C, G, _, or. . However,
 * this program does require (and checks) that these alignments are
 * all the same length.
 *
 * This program attempts to compute the distance between them by
 * counting how many of those match. This is done by onehot encoding
 * them, storing them as bitfields, and then doing bitwise and (&&) of the
 * result and counting the matches. The A, T, C, G, and "other" are
 * encoded in fields of 0x10000, 0x01000, 0x00100, 0x00010, and
 * 0x00001 and pushed into 32 bit words for each species.
 *
 * These are then used to compute distance by counting the number of
 * times they have 1 in the same place (which is matches) and
 * subtracting that from the overall alignment length.
 */
#include <stdio.h>
#include <stdlib.h>

/* 
 * This is the maximum number of species we can have: the file has
 * like 117k of them 
 */
#define MAX_SPECIES (128 * 1024)

/* 
 * And this is the maximum number of pairs in the aligned gene: the
 * alignments are about 7681 long. Give enough padding to keep things
 * clean.
 */
#define MAX_ALIGNMENT_LENGTH (8 * 1024)
#define MAX_WORDS (8*MAX_ALIGNMENT_LENGTH/32)

/* The lines can be REALLY long */
static int num_species = -1;
static char line[MAX_ALIGNMENT_LENGTH];

/* 
 * This holds the data for a species, specifically the name and the
 * onehots coding of the alignment. These should all have the same
 * number of words.
 */
typedef struct {
  char name[128];
  int num_words;
  int words[MAX_WORDS];
} SPECIES;

/* Statically declare space for all the species */
static SPECIES species[MAX_SPECIES];

/* The main and only routine that does everything described in the header */
int main(int argc, char **argv)
{

  char *pname,*inname, *outname;

  /* Character pointers used to parse lines from input to output */
  char *in, *out;

  /* Counters and limits */
  int ibit, iword, num_bits;

  /* 
   * This holds the total length of the alignment, which must be the
   * same for all species in the file.
   */
  int alignment_length;

  /* And the distance */
  int distance;
  
  /* Used to construct the bit-field words */
  int onehots, current_word;

  /* 
   * This one I use to display a progress bar in when computing the
   * distance matrix 
   */
  int num_distances;
  
  /* A file pointer used for both input and output */
  FILE *fp;

  /* 
   * Pointer to current species, used for parsing the input, and the
   * "first" and "second" species, used for computing distance
   */
  SPECIES *current_species = species;
  SPECIES *first, *second;

  if (argc != 3) {
    printf(" Usage: createLowerDistanceMatrix InputFileName OutputFileName\n");
    exit(1);
  }

  /* Open the input file */
  if ((fp = fopen(argv[1],"r")) == NULL) {
    printf(" Can't open <%s>\n",argv[1]);
    exit(1);
  }

  /* 
   * Parse lines: this is written this way to allow the alignments to
   * be broken amongst lines if necessary
   */
  printf(" Parse <%s> ... \n",argv[1]);
  while(fgets(line,sizeof(line),fp) != NULL) {

    if (line[0] == '>') {

      /* 
       * We have a new species header line. If this is NOT the first
       * one, that means we just finished parsing a species. Store
       * what is left of the word in the last word for the species
       */
      if (num_species >= 0) {

	/* Add a word */
	if (current_species->num_words < MAX_WORDS) {
	  current_word = current_word << (32 - num_bits);
	  current_species->words[current_species->num_words++] = current_word;
	} else {
	  printf("Error: Too many alignment words\n");
	  exit(1);
	}

	/* 
	 * If this is the first species, store the number of
	 * alignments, otherwise, check the number of alignments
	 * against the prior saved value and if it doesn't match, we
	 * choose to ignore it. NOTE: this all makes the standard
	 * assumptions about integer math.
	 */
	if (num_species == 0) {
	  alignment_length = (32 * (current_species->num_words-1) + num_bits)/5;
	  printf(" .... Alignment length is %d\n",alignment_length);
	} else if (num_species > 0) {
	  if ((32*(current_species->num_words-1) + num_bits)/5 != alignment_length){
	    printf(" Error: Species[%d] <%s> has strange aligment (%d,%d) : ignore\n",
		   (int)(current_species - species),
		   current_species->name,
		   alignment_length,
		   (32*current_species->num_words + num_bits)/5);

	    num_species--;
	  }
	}
      }
      
      /* It's a new species: go to the next one */
      if (++num_species == MAX_SPECIES) break;
      current_species = species + num_species;
      current_species->num_words = 0;

      /* Skip past the first word */
      for (in = line; *in != ' ' && *in != 0; in++)
	;
      in++;

      /* Copy characters into the name of the current species */
      out = current_species->name;
      while (*in != '\n' && *in != 0) *out++ = *in++;
      *out = 0;

      num_bits = 0;
      current_word = 0;
      current_species->num_words = 0;

    } else {

      /* 
       * Parse the characters in the line: first form the onehots for
       * this character 
       */
      for (in = line; *in != '\n' && *in != 0; in++) {
	switch (*in) {
	case 'A':
	  onehots = 0x10;
	  break;
	case 'G':
	  onehots = 0x8;
	  break;
	case 'T':
	  onehots = 0x4;
	  break;
	case 'C':
	  onehots = 0x2;
	  break;
	default:
	  onehots = 0x1;
	  break;
	}

	/* Now push the one hots onto the current word ... */
	for (ibit=0;ibit<5;ibit++) {
	  current_word = current_word << 1;
	  current_word |= (onehots & 0x1);
	  onehots = onehots >> 1;
	  /* 
	   * If we have pushed all the bits, store this word and make
	   * a new one
	   */
	  if (++num_bits == 32) {
	    if (current_species->num_words < MAX_WORDS) {
	      current_species->words[current_species->num_words++] =
		current_word;
	    } else {
	      printf("Error: Too many alignment words\n");
	      exit(1);
	    }
	    current_word = num_bits = 0;
	  }
	}
	
      }
    }
  }
  printf(" Read (%d) Species\n",num_species);
  fclose(fp);
  
  /* 
   * Ok, now we have all the data loaded we think.  Let go through a
   * double loop and compute the hamming distance, but first open the
   * file
   */
  {
    long long temp = num_species;
    temp = (temp * (temp+1)) / 2;
    printf(" Write file <%s> with (%lld distances = %lld lines of dots)\n",
	   argv[2],temp,temp/70000000);
  }
  if ((fp = fopen(argv[2],"w")) == NULL) {
    printf(" Error: Could not open <%s>\n",argv[2]);
    exit(1);
  }

  /* 
   * This is a double loop over the lower diaganal using pointer math
   * for speed (not that it really matters)
   */
  num_distances = 0;
  for (first = species; first - species < num_species;first++) {
    for (second = species; second - species < first - species; second++) {

      /* 
       * We know they have the same number of words from before. What
       * this loop does is go through and find the number of times the
       * onehots are equal. 
       */
      distance = 0;
      for (iword = 0; iword < first->num_words; iword++) {
	current_word = first->words[iword] & second->words[iword];
	for (ibit = 0; ibit < 32; ibit++) {
	  distance += current_word & 0x1;
	  current_word = current_word >> 1;
	}
      }

      /* 
       * The distance is actually the total alignment length minus the
       * number of matches 
       */
      distance = alignment_length - distance;
      fprintf(fp,"%d,",distance);

      if ((++num_distances % 1000000) == 0) {
	printf(".");
	fflush(stdout);
      }
      if (num_distances % 70000000 == 0) printf("\n");
    }
    fprintf(fp,"\n");
  }

  /* Close the file */
  fclose(fp);
  printf("DONE!!!\n");
}
