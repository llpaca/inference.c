package main

// import (
// 	"encoding/csv"
// 	"fmt"
// 	"io"
// 	"os"
// 	"strconv"
// )

// type Post struct {
// 	QID      int
// 	Question string
// 	Author   string
// 	AuthorID int
// 	Answer   string
// }

// func main() {
// 	file, err := os.Open("mathstack-qa/data.txt")
// 	if err != nil {
// 		fmt.Println("Error opening file:", err)
// 		return
// 	}
// 	defer file.Close()

// 	reader := csv.NewReader(file)
// 	reader.Comma = '\t'

// 	// Read header row
// 	header, err := reader.Read()
// 	if err != nil {
// 		fmt.Println("Error reading header:", err)
// 		return
// 	}
// 	fmt.Println("Header:", header)

// 	var posts []Post
// 	rowNum := 1

// 	for {
// 		record, err := reader.Read()
// 		if err == io.EOF {
// 			break
// 		}
// 		if err != nil {
// 			fmt.Printf("Error reading row %d: %v\n", rowNum, err)
// 			continue
// 		}
// 		rowNum++

// 		if len(record) != 5 {
// 			fmt.Printf("Skipping row %d: expected 5 fields, got %d\n", rowNum, len(record))
// 			continue
// 		}

// 		qid, err := strconv.Atoi(record[0])
// 		if err != nil {
// 			fmt.Printf("Skipping row %d: invalid qid %q: %v\n", rowNum, record[0], err)
// 			continue
// 		}

// 		authorID, err := strconv.Atoi(record[3])
// 		if err != nil {
// 			fmt.Printf("Row %d: invalid author_id %q, defaulting to 0\n", rowNum, record[3])
// 			authorID = 0
// 		}

// 		posts = append(posts, Post{
// 			QID:      qid,
// 			Question: record[1],
// 			Author:   record[2],
// 			AuthorID: authorID,
// 			Answer:   record[4],
// 		})
// 	}

// 	fmt.Printf("\nParsed %d posts total\n", len(posts))
// 	if len(posts) > 0 {
// 		for i := range(5){
// 			fmt.Printf("post: %d\n  QID: %d\n  Question: %.80s...\n  Author: %s\n  AuthorID: %d\n  Answer: %.80s...\n",
// 				i,posts[i].QID, posts[i].Question, posts[i].Author, posts[i].AuthorID, posts[i].Answer)
// 		}
// 	}
// }