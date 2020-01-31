#include <stdio.h>
#include <string.h>

#define LIMIT 10000
#define LINE_SIZE 128

//For part 1
void assemble(char[]);
int findOrigin(FILE*);
int firstPass(FILE*, int[], int);
void printLabels(int[]);

//For part 2
int getAdd(char[]);
int getAnd(char[]);
int getTrap(char[]);
int getNot(char[]);
int getLd(char[], int[], int);
int getLdr(char[]);
int getSt(char[], int[], int);
int getStr(char[]);
int getBr(char[], int[], int);
int getIt(int, char[]);
int secondPass(FILE*, int[], int);

char* scan(char* word)
{
	int j = 0; 
	for (int i = 0; word[i] != 0; ++i)
	{
		if (word[i] != 32 && word[i] != 9) 
		{
			word[j] = word[i];
			j++;
		}
	}
	word[j] = '\0';
	return word;
}
void toUpper(char* word)
{
	for (int i = 0; word[i] != 0; ++i)
	{
		if ( word[i] >= 'a' && word[i] <= 'z') word[i] = word[i] - 32;
	}
}
void printLabels(int labels[])
{
	printf("labels = {%d, %d, %d, %d, %d, %d, %d, %d, %d, %d}\n", labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6], labels[7], labels[8], labels[9]);
}

void assemble(char filename[])
{
	//Open the file for reading
    FILE *infile = fopen( filename , "r" );
    
	if (infile != NULL) 
	{    
		//Create labels array and set all elements to -1.  
		int labels[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
		
		int lc = findOrigin(infile);
		if (lc > -1)
		{
			//Read in label values
			if (!firstPass(infile, labels, lc))
			{
				//Show the labels.
				//printLabels(labels);
				//The following is for part 2
				rewind(infile);
				secondPass(infile, labels, lc);
			}
		}
		
		//Close the file
		fclose(infile);
		
	} 
	else 
	{	
		printf("Can't open input file.\n");		
	}    

}


int findOrigin(FILE *infile)
{
	//Each trip through the while loop will read the next line of infile
	//into the line char array as a null terminated string.
	char line[LINE_SIZE]; 
	int result = -1;
	//The variable lineCount keeps track of how many lines have been read.
	//It is used to guard against infinite loops.  Don't remove anything
	//that has to do with linecount.
	int lineCount = 0; //
	
	//For getting out of the while loop.
	int done = 0;
	
	//For getting rid of the trailing newline
	char c;
	//Read lines until EOF reached or LIMIT lines read.  Limit prevent infinite loop.
	//while (fscanf(infile, "%[^\n]s", line) != EOF && lineCount < LIMIT && !done)
	while (!done && lineCount < LIMIT && fscanf(infile, "%[^\n]s", line) != EOF)
	{
		lineCount++;
		fscanf(infile, "%c", &c);  //Get rid of extra newline.
		
		//At this point, line contains the next line of the ASM file.
		//Put your code here for seeing if line is an origin.
		//Options:
		//	1. line is an origin (save value, set done = 1).  
		char* ch = scan(line);
		toUpper(ch);
		if (strncmp(ch, ".ORIG", 5) == 0)
		{
			ch += 5;
			if (strncmp(ch, "X", 1) == 0)
			{
				ch++;
				sscanf(ch, "%X", &result);
				done = 1;
			}
			else
			{
				sscanf(ch, "%d", &result);
				done = 1;
			}
			
		}
		//  2. line is a blank line (skip).
		//  3. line is a comment (skip).
		else if (strncmp(ch, ";", 1) == 0 || strncmp(ch, "", 1) == 0)
		{}
		//  4. line is anything else (print error, set done = 1).
		else
		{
			printf("ERROR 1: Missing origin directive. Origin must be first line in program. \n");
			done = 1;
		}
		//Set the line to empty string for the next pass.
		line[0] = 0;
	}
	
	
	//At this point you must decide if an origin was found or not.
	//How you do this is up to you.
	//If a good origin was found, check the size.  
	//		if it is too big, print an error and return -1.
	//      if it is not too big, return the value.
	//If a good origin was NOT found, print the error message and return -1.
	if (result > 0xffff)
	{
		printf("ERROR 2: Bad origin address. Address too big for 16 bits. \n");
		result = -1;
	}
	else if (result == -1)
	{
		printf("ERROR 1: Missing origin directive. Origin must be first line in program. \n");
	}
	return result; 
}

int getAdd(char ch[])
{
	ch = scan(ch);
	int result = 0b0001;
	result <<= 3;
	int temp = ch[4] - 48;
	result = result | temp;
	result <<= 3;
	temp = ch[7] - 48;
	result = result | temp;
	result <<= 3; 
	if (strncmp(&ch[9], "#", 1) == 0)
	{
		result = result | 0b100;
		result <<= 3;
	 	sscanf(&ch[10], "%d", &temp);
		int conv = 0b011111;
		temp &= conv;
		result = result | temp;
	}
	else
	{
		result <<= 3;
		temp = ch[10] - 48;
		result = result + temp;
	}
	
	return result;
}

int getAnd(char ch[])
{
	ch = scan(ch);
	int result = 0b0101;
	result <<= 3;
	int temp = ch[4] - 48;
	result = result | temp;
	result <<= 3;
	temp = ch[7] - 48;
	result = result | temp;
	result <<= 3; 
	if (strncmp(&ch[9], "#", 1) == 0)
	{
		result = result | 0b100;
		result <<= 3;
	 	sscanf(&ch[10], "%d", &temp);
		int conv = 0b011111;
		temp &= conv;
		result = result | temp;
	}
	else
	{
		result <<= 3;
		temp = ch[10] - 48;
		result = result + temp;
	}
	
	return result;
}

int getTrap(char ch[])
{
	ch = scan(ch);
	unsigned int result = 0b011110000;
	result <<= 8;
	int temp = 0;
	sscanf(&ch[5], "%X", &temp);
	result = result | temp;
	return result;
}

int getNot(char ch[])
{
	ch = scan(ch);
	int result = 0b1001;
	result <<= 3;
	int temp = ch[4] - 48;
	result = result | temp;
	result <<= 3;
	temp = ch[7] - 48;
	result = result | temp;
	result <<= 6; 
	result = result | 0b111111;
	return result; 
}

int getLd(char ch[], int labels[], int lc)
{
	ch = scan(ch);
	int result = 0b0010;
	result <<= 3;
	int temp = ch[3] - 48;
	result = result | temp;
	result <<= 9; 
	temp = ch[6] - 48;
	int label = labels[temp];
	label = label - lc;
	label = label & 0b0000000111111111;
	result = result | label;
	return result;
}

int getLdr(char ch[])
{
	ch = scan(ch);
	unsigned int result = 0b0110;
	result <<= 3;
	int temp = ch[4] - 48;
	result = result | temp;
	result <<= 3;
	temp = ch[7] - 48;
	result = result | temp;
	result <<= 6;
	sscanf(&ch[10], "%d", &temp);
	int conv = 0b0111111;
	temp &= conv;
	result = result | temp;
	return result;
}

int getSt(char ch[], int labels[], int lc)
{
	ch = scan(ch);
	int result = 0b0011;
	result <<= 3;
	int temp = ch[3] - 48;
	result = result | temp;
	result <<= 9; 
	temp = ch[6] - 48;
	int label = labels[temp];
	label = label - lc;
	label = label & 0b0000000111111111;
	result = result | label;
	return result;
}

int getStr(char ch[])
{
	ch = scan(ch);
	unsigned int result = 0b0111;
	result <<= 3;
	int temp = ch[4] - 48;
	result = result | temp;
	result <<= 3;
	temp = ch[7] - 48;
	result = result | temp;
	result <<= 6;
	sscanf(&ch[10], "%d", &temp);
	int conv = 0b0111111;
	temp &= conv;
	result = result | temp;
	return result;
}

int getBr(char ch[], int labels[], int lc)
{
	ch = scan(ch);
	int result = 0b0000;
	result <<= 3;
	int i = 2;
	int n = 0;
	int z = 0;
	int p = 0;
	while(strncmp(&ch[i], "L", 1) != 0)
	{
		if (strncmp(&ch[i], "N", 1) == 0 && !z && !p)
		{
			result = result | 0b100;
			n = 1;
		}
		else if(strncmp(&ch[i], "Z", 1) == 0 && !p)
		{
			result = result | 0b010;
			z = 1;
		}
		else if(strncmp(&ch[i], "P", 1) == 0)
		{
			result = result | 0b001;
			z = 1;
			p = 1;
		}
		i++;
	}
	i++;
	if (!n && !z && !p)
		result = result | 0b111;
	result <<= 9; 
	int temp = ch[i] - 48;
	int label = labels[temp];
	label = label - lc;
	label = label & 0b0000000111111111;
	result = result | label;
	return result;
}

int firstPass(FILE *infile, int labels[], int lc)
{
	//Create a while loop similar to findOrigin.
	//You can rewind if you want to but you shouldn't have to.
	
	//Read a line.
		//If the line is a comment, a blank line or the .orig directive, don�t do anything.
		//If the line is a label on a line by itself, save the lc to the labels array at the appropriate position.
		//	For example, if L3, store the lc value in labels[3].
		//If the line is a label followed by .fill, save the lc in labels AND increment lc.
		//If the line is .end, return 0 for done with no error.
		//If the end of file is reached before .end is found print the error and return -1.
		//If the line is one of the allowed instructions (ADD, AND, NOT, LD, LDR, ST, STR, BR, and TRAP) increment the lc.
		//If the line is anything else print the unknown instruction error and return -1.
		//Each trip through the while loop will read the next line of infile
	
	//into the line char array as a null terminated string.
	char line[LINE_SIZE]; 
	int result = -1;
	//The variable lineCount keeps track of how many lines have been read.
	//It is used to guard against infinite loops.  Don't remove anything
	//that has to do with linecount.
	int lineCount = 0; //
	line[0] = 0;
	//For getting out of the while loop.
	int done = 0;
	
	//For getting rid of the trailing newline
	char c;
	//Read lines until EOF reached or LIMIT lines read.  Limit prevent infinite loop.
	//while (fscanf(infile, "%[^\n]s", line) != EOF && lineCount < LIMIT && !done)
	while (!done && lineCount < LIMIT && fscanf(infile, "%[^\n]s", line) != EOF)
	{
		lineCount++;
		fscanf(infile, "%c", &c);  //Get rid of extra newline.
		char* cmp = scan(line);
		toUpper(cmp);
		if (strncmp(cmp, "L", 1) == 0 && strncmp(cmp, "LDR", 3) != 0 && strncmp(cmp, "LD", 2) != 0)
		{
			int index = cmp[1] - 48;
			cmp += 2;
			if (strncmp(cmp, ".FILL", 5) == 0)
			{
				labels[index] = lc; 
				lc++;
			}
			else
				labels[index] = lc;
			
		}
		else if (strncmp(cmp, ";", 1) == 0 || strncmp(cmp, "", 1) == 0 || strncmp(cmp, ".ORIG", 5) == 0)
		{

		}
		else if (strncmp(cmp, ".END", 4) == 0)
		{
			done = 1;
			result = 0;
		}
		else if (strncmp(cmp, "ADD", 3) == 0 || strncmp(cmp, "AND", 3) == 0 || strncmp(cmp, "NOT", 3) == 0 || strncmp(cmp, "LDR", 3) == 0 || strncmp(cmp, "STR", 3) == 0
		|| strncmp(cmp, "LD", 2) == 0 || strncmp(cmp, "ST", 2) == 0 || strncmp(cmp, "BR", 2) == 0 || strncmp(cmp, "TRAP", 4) == 0 ||  strncmp(cmp, "BRZ", 3) == 0
		|| strncmp(cmp, "BRN", 3) == 0 || strncmp(cmp, "BRP", 3) == 0 || strncmp(cmp, "BRNZ", 4) == 0 || strncmp(cmp, "BRZP", 4) == 0 || strncmp(cmp, "BRNP", 4) == 0
		|| strncmp(cmp, "BRNZP", 5) == 0)
		{
			//when submitting if you get a problem with error 3 add in all permutations of branch. 
			lc++;
		}
		else
		{
			printf("%s\n", "ERROR 3: Unknown instruction.");
			done = 1;
		}
		line[0] = 0;
	}
	if (!done)
	{
		printf("%s\n", "ERROR 4: Missing end directive.");
		return -1;
	}
	return result;
}

int secondPass(FILE *infile, int labels[], int lc)
{
	char line[LINE_SIZE]; 
	int result = -1;
	//The variable lineCount keeps track of how many lines have been read.
	//It is used to guard against infinite loops.  Don't remove anything
	//that has to do with linecount.
	int lineCount = 0;
	line[0] = 0;
	//For getting out of the while loop.
	int done = 0;
	
	//For getting rid of the trailing newline
	char c;
	//Read lines until EOF reached or LIMIT lines read.  Limit prevent infinite loop.
	//while (fscanf(infile, "%[^\n]s", line) != EOF && lineCount < LIMIT && !done)
	printf("%04X\n", lc);
	while (!done && lineCount < LIMIT && fscanf(infile, "%[^\n]s", line) != EOF)
	{
		lineCount++;
		fscanf(infile, "%c", &c);  //Get rid of extra newline.

		toUpper(line);
		char* ch = scan(line);
		if (strncmp(ch, "ADD", 3) == 0)
		{
			lc++;
			printf("%04X\n", getAdd(ch));
		}
		else if(strncmp(ch, "AND", 3) == 0)
		{
			lc++;
			printf("%04X\n", getAnd(ch));
		}
		else if(strncmp(ch, "TRAP", 4) == 0)
		{
			lc++;
			printf("%04X\n", getTrap(ch));
		}
		else if(strncmp(ch, "NOT", 3) == 0)
		{
			lc++;
			printf("%04X\n", getNot(ch));
		}
		else if(strncmp(ch, "LDRR", 4) == 0)
		{
			lc++;
			printf("%04X\n", getLdr(ch));
		}
		else if(strncmp(ch, "LD", 2) == 0)
		{
			lc++;
			printf("%04X\n", getLd(ch, labels, lc));
		}
		else if(strncmp(ch, "STRR", 4) == 0)
		{
			lc++;
			printf("%04X\n", getStr(ch));
		}
		else if(strncmp(ch, "ST", 2) == 0)
		{
			lc++;
			printf("%04X\n", getSt(ch, labels, lc));
		}
		else if(strncmp(ch, "BR", 2) == 0)
		{
			lc++;
			printf("%04X\n", getBr(ch, labels, lc));
		}
		else if(strncmp(ch, ".END", 4) == 0)
		{
			done = 1;
			result = 0;
		}
		else if(strncmp(ch, "L", 1) == 0 && strncmp(ch, "LDR", 3) != 0 && strncmp(ch, "LD", 2) != 0)
		{
			ch += 2;
			ch = scan(ch);
			if (strncmp(ch, ".FILL", 5) == 0)
			{
				lc++;
				printf("%04X\n", 0000); 
			}
		}
		line[0] = 0; 
	}
	return result;
}

