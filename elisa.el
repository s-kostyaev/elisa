;;; elisa.el --- Emacs Lisp Information System Assistant -*- lexical-binding: t -*-

;; Copyright (C) 2024, 2025 Free Software Foundation, Inc.

;; Author: Sergey Kostyaev <sskostyaev@gmail.com>
;; URL: http://github.com/s-kostyaev/elisa
;; Keywords: help local tools
;; Package-Requires: ((emacs "29.2") (ellama "0.11.2") (llm "0.18.1") (async "1.9.8") (plz "0.9"))
;; Version: 1.1.7
;; SPDX-License-Identifier: GPL-3.0-or-later
;; Created: 18th Feb 2024

;; This file is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation; either version 3, or (at your option)
;; any later version.

;; This file is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License
;; along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.

;;; Commentary:
;;
;; ELISA (Emacs Lisp Information System Assistant) is a system
;; designed to generate informative answers to user queries using a
;; Retrieval Augmented Generation (RAG) approach.  RAG combines the
;; capabilities of Large Language Models (LLMs) with Information
;; Retrieval (IR) techniques to enhance the accuracy and relevance of
;; generated responses.
;;
;; ELISA addresses limitations inherent in purely LLM-based systems by:
;;
;; - Leveraging External Knowledge: Unlike LLMs trained on a fixed
;; dataset, ELISA can access and process information from external
;; knowledge sources, expanding its knowledge base beyond its initial
;; training data.
;;
;; - Reducing Computational Requirements: Instead of
;; retraining the entire LLM for new information, ELISA focuses on
;; retrieving relevant data, minimizing computational resources
;; required for query processing.
;;
;; - Minimizing Hallucinations: By grounding responses in factual
;; data retrieved from external sources, ELISA aims to reduce the
;; likelihood of generating incorrect or nonsensical information
;; (hallucinations) often associated with LLMs.
;;
;; The following sections will detail the key components and processes
;; involved in ELISA's operation: parsing, retrieval, augmentation,
;; and generation.
;;
;; Parsing.
;;
;; Simple solution is split text document into chunks by length with
;; overlap and save it to storage.  In ELISA we use more advanced
;; solution.  Instead of split by length we split text by semantic
;; distances between parts of text document.  To store this chunks we
;; use sqlite database.
;;
;; Retrieving.
;;
;; Simple solution is to extract top K chunks by semantic similarity
;; from storage.  To improve quality we use more robust solution.
;;
;; Before going to storage we let LLM rewrite user query to make it
;; context agnostic.  For example, user ask LLM about llamas, LLM
;; answer.  Then user ask "where it lives?".  If we try to search for
;; this query we barely find something useful.  But LLM can rewrite it
;; to something like "where llamas lives?" and we will find useful
;; information.
;;
;; Instead of use simple semantic similarity search only we use hybrid
;; search.  It means that ELISA search relevant chunks by semantic
;; similarity and by full text search and then combine results.
;;
;; To improve relevance even more instead of use top K results from
;; hybrid search user can enable reranker.  Reranker is a service that
;; gives user query, top N chunks and feed it to reranker model.  This
;; model gives pairs of text: user query and text chunk and return
;; number that means relevance.  Service collect this results and sort
;; it by relevance.  Also if reranker enabled ELISA filter out
;; irrelevant results.
;;
;; Augmentation.
;;
;; ELISA gets text retrieved in previous step and put it into context.
;; This context later will be sent to LLM together with user query.
;;
;; Generation.
;;
;; To improve generation quality to user query will be added
;; instructions to LLM how to answer.  We let LLM ability to say "not
;; enough data" instead of hallucinations.  LLM generates answer based
;; on context, instructions and user query.

;;; Code:
(require 'ellama)
(require 'llm)
(require 'llm-provider-utils)
(require 'info)
(require 'async)
(require 'dom)
(require 'shr)
(require 'plz)
(require 'json)
(require 'sqlite)

(defgroup elisa nil
  "RAG implementation for `ellama'."
  :group 'tools)

(defcustom elisa-embeddings-provider (progn (require 'llm-ollama)
					    (make-llm-ollama
					     :embedding-model "nomic-embed-text"))
  "Embeddings provider to generate embeddings."
  :type '(sexp :validate llm-standard-provider-p))

(defcustom elisa-chat-provider (progn (require 'llm-ollama)
				      (make-llm-ollama
				       :chat-model "sskostyaev/openchat:8k-rag"
				       :embedding-model "nomic-embed-text"))
  "Chat provider."
  :type '(sexp :validate llm-standard-provider-p))

(defcustom elisa-db-directory (file-truename
			       (file-name-concat
				user-emacs-directory "elisa"))
  "Directory for elisa database."
  :type 'directory)

(defcustom elisa-limit 5
  "Count quotes to pass into llm context for answer."
  :type 'natnum)

(defcustom elisa-find-executable find-program
  "Path to find executable."
  :type 'string)

(defcustom elisa-tar-executable "tar"
  "Path to tar executable."
  :type 'string)

(defcustom elisa-sqlite-vss-version "v0.1.2"
  "Sqlite VSS version."
  :type 'string)

(defcustom elisa-sqlite-vss-path nil
  "Path to sqlite-vss extension."
  :type 'file)

(defcustom elisa-sqlite-vector-path nil
  "Path to sqlite-vector extension."
  :type 'file)

(defcustom elisa-semantic-split-function #'elisa-split-by-paragraph
  "Function for semantic text split."
  :type 'function)

(defcustom elisa-prompt-rewriting-enabled t
  "Enable prompt rewriting for better retrieving."
  :type 'boolean)

(defcustom elisa-chat-prompt-template
  "Answer user query based on context above. \
If you can answer it partially do it. \
Provide list of open questions if any. \
Say \"not enough data\" if you can't answer user \
query based on provided context. User query:
%s"
  "Chat prompt template.
Contains instructions to LLM to be more focused on data in
context, be able to say \"I don't know\" etc. User query will be
inserted at the end and all this result prompt will be sent to
LLM together with context."
  :type 'string)

(defcustom elisa-rewrite-prompt-template
  "<INSTRUCTIONS>
You are professional search agent. With given context and user
prompt you need to create new prompt for search **IN THE SAME
LANGUAGE AS ORIGINAL USER PROMPT**. It should be concise and
useful without additional context. Response with prompt only. You
should replace all words like 'this' or 'it' to its values to
make search successful. If user prompt contains question your
prompt should also be in form of question.
 </INSTRUCTIONS>
<EXAMPLE>
 - What is pony?
 - Pony is ...
 - How to buy it?

How to buy a pony?
</EXAMPLE>
<USER_PROMPT>
%s
</USER_PROMPT>"
  "Prompt template for prompt rewriting."
  :type 'string)

(defcustom elisa-research-topics-generator-template
  "<INSTRUCTIONS>
You are professional researcher. User will provide you a theme
for research. You need to generate list of topics for deeper
research and wider theme coverage.
</INSTRUCTIONS>
<THEME>
%s
</THEME>"
  "Prompt template for research topics generation."
  :type 'string)

(defcustom elisa-research-questions-generator-template
  "<INSTRUCTIONS>
You are professional researcher. User will provide you a theme
and a topic for research. You need to generate list of questions
for search to cover this topic. Focus on topic.
</INSTRUCTIONS>
<THEME>
%s
</THEME>
<TOPIC>
%s
</TOPIC>"
  "Prompt template for research questions generation."
  :type 'string)

(defcustom elisa-tika-url "http://localhost:9998/"
  "Apache tika url for file parsing."
  :type 'string)

(defcustom elisa-searxng-url "http://localhost:8080/"
  "Searxng url for web search.  Json format should be enabled for this instance."
  :type 'string)

(defcustom elisa-pandoc-executable "pandoc"
  "Path to pandoc (https://pandoc.org/) executable."
  :type 'string)

(defcustom elisa-webpage-extraction-function #'elisa-get-webpage-buffer
  "Function to get buffer with webpage content."
  :type 'function)

(defcustom elisa-complex-file-extraction-function #'elisa-parse-with-tika-buffer
  "Function to get buffer with complex file (like pdf, odt etc.) content."
  :type 'function)

(defcustom elisa-web-search-function #'elisa-search-duckduckgo
  "Function to search the web.
Function should get prompt and return list of urls."
  :type 'function)

(defcustom elisa-web-pages-limit 10
  "Limit of web pages to parse during web search."
  :type 'natnum)

(defcustom elisa-breakpoint-threshold-amount 0.4
  "Breakpoint threshold amount.
Increase it if you need decrease semantic split granularity."
  :type 'number)

(defcustom elisa-reranker-enabled nil
  "Enable reranker to improve retrieving quality.
Reranker is a service to improve answer quality by mesure
relevance of text chunks to user query and sort chunks by
relevance.  See https://github.com/s-kostyaev/reranker for more
details."
  :type 'boolean)

(defcustom elisa-reranker-url "http://127.0.0.1:8787/"
  "Reranker service url.
Reranker is a service to improve answer quality by mesure
relevance of text chunks to user query and sort chunks by
relevance.  See https://github.com/s-kostyaev/reranker for more
details."
  :type 'string)

(defcustom elisa-reranker-similarity-threshold 0
  "Reranker similarity threshold.
If set, all quotes with similarity less than threshold will be filtered out."
  :type 'number)

(defcustom elisa-reranker-limit 20
  "Number of quotes for send to reranker."
  :type 'integer)

(defcustom elisa-ignore-patterns-files '(".gitignore" ".ignore" ".rgignore")
  "Files with patterns to ignore during file parsing."
  :type '(repeat string))

(defcustom elisa-ignore-invisible-files t
  "Ignore invisible files and directories during file parsing."
  :type 'boolean)

(defcustom elisa-enabled-collections '("builtin manuals" "external manuals")
  "Enabled collections for elisa chat."
  :type '(repeat string))

(defcustom elisa-supported-complex-document-extensions '("doc" "dot" "ppt" "xls" "rtf" "docx" "pptx" "xlsx" "xlsm" "pdf" "epub" "msg" "odt" "odp" "ods" "odg" "docm")
  "Supported complex document file extensions."
  :type '(repeat string))

(defcustom elisa-batch-embeddings-enabled nil
  "Enable batch embeddings if supported."
  :type 'boolean)

(defcustom elisa-batch-size 300
  "Batch size to send to provider during batch embeddings calculation."
  :type 'integer)

(defun elisa-supported-complex-document-p (path)
  "Check if PATH contain supported complex document."
  (cl-find (file-name-extension path)
	   elisa-supported-complex-document-extensions :test #'string=))

(defun elisa-sqlite-vss-download-url ()
  "Generate sqlite vss download url based on current system.
Sqlite vss is an extension to sqlite providing vector search
similarity support that used to retrieve relevant data from
database."
  (cond  ((eq system-type 'darwin)
	  (if (or (string-prefix-p "aarch64" system-configuration)
		  (string-prefix-p "arm" system-configuration))
	      (format
	       "https://github.com/asg017/sqlite-vss/releases/download/%s/sqlite-vss-%s-loadable-macos-aarch64.tar.gz"
	       elisa-sqlite-vss-version
	       elisa-sqlite-vss-version)
	    (format
	     "https://github.com/asg017/sqlite-vss/releases/download/%s/sqlite-vss-%s-loadable-macos-x86_64.tar.gz"
	     elisa-sqlite-vss-version
	     elisa-sqlite-vss-version)))
	 ((eq system-type 'gnu/linux)
	  (format
	   "https://github.com/asg017/sqlite-vss/releases/download/%s/sqlite-vss-%s-loadable-linux-x86_64.tar.gz"
	   elisa-sqlite-vss-version
	   elisa-sqlite-vss-version))
	 (t (user-error "Can't determine download url"))))

(defun elisa--vss-path ()
  "Path to vss sqlite extension."
  (or elisa-sqlite-vss-path
      (let* ((ext (if (eq system-type 'darwin) "dylib" "so"))
	     (file (format "vss0.%s" ext)))
	(file-name-concat elisa-db-directory file))))

(defun elisa--vector-path ()
  "Path to vector sqlite extension."
  (or elisa-sqlite-vector-path
      (let* ((ext (if (string-equal system-type 'darwin) "dylib" "so"))
	     (file (format "vector0.%s" ext)))
	(file-name-concat elisa-db-directory file))))

;;;###autoload
(defun elisa-download-sqlite-vss ()
  "Download sqlite vss."
  (interactive)
  (let ((file-name
	 (file-truename
	  (file-name-concat
	   elisa-db-directory
	   (format "sqlite-vss-%s.tar.gz" elisa-sqlite-vss-version))))
	(default-directory elisa-db-directory))
    (make-directory elisa-db-directory t)
    (url-copy-file (elisa-sqlite-vss-download-url) file-name t)
    (process-lines (executable-find elisa-tar-executable) "-xf" file-name)
    (delete-file file-name))
  (elisa--reopen-db))

(defun elisa-get-embedding-size ()
  "Get embedding size."
  (length (llm-embedding elisa-embeddings-provider "test")))

(defun elisa-embeddings-create-table-sql ()
  "Generate sql for create embeddings table."
  "DROP TABLE IF EXISTS elisa_embeddings;")

(defun elisa-data-embeddings-create-table-sql ()
  "Generate sql for create data embeddings table."
  (format "CREATE VIRTUAL TABLE IF NOT EXISTS data_embeddings USING vss0(embedding(%d));"
	  (elisa-get-embedding-size)))

(defun elisa-data-embeddings-drop-table-sql ()
  "Generate sql for drop data embeddings table."
  "DROP TABLE IF EXISTS data_embeddings;")

(defun elisa-data-fts-create-table-sql ()
  "Generate sql for create full text search table."
  "CREATE VIRTUAL TABLE IF NOT EXISTS data_fts USING FTS5(data);")

(defun elisa-info-create-table-sql ()
  "Generate sql for create info table."
  "DROP TABLE IF EXISTS info;")

(defun elisa-collections-create-table-sql ()
  "Generate sql for create collections table."
  "CREATE TABLE IF NOT EXISTS collections (name TEXT UNIQUE);")

(defun elisa-kinds-create-table-sql ()
  "Generate sql for create kinds table."
  "CREATE TABLE IF NOT EXISTS kinds (name TEXT UNIQUE);")

(defun elisa-fill-kinds-sql ()
  "Generate sql for fill kinds table."
  "INSERT INTO KINDS (name) VALUES ('web'), ('file'), ('info') ON CONFLICT DO NOTHING;")

(defun elisa-files-create-table-sql ()
  "Generate sql for create files table."
  "CREATE TABLE IF NOT EXISTS files (path TEXT UNIQUE, hash TEXT)")

(defun elisa-data-create-table-sql ()
  "Generate sql for create data table."
  "CREATE TABLE IF NOT EXISTS data (
kind_id INTEGER,
collection_id INTEGER,
path TEXT,
hash TEXT,
data TEXT,
FOREIGN KEY(kind_id) REFERENCES kinds(rowid),
FOREIGN KEY(collection_id) REFERENCES collections(rowid)
);")

(defun elisa--init-db (db)
  "Initialize elisa DB."
  (if (not (file-exists-p (elisa--vss-path)))
      (warn "Please run M-x `elisa-download-sqlite-vss' to use this package")
    (sqlite-pragma db "PRAGMA journal_mode=WAL;")
    (sqlite-load-extension db (elisa--vector-path))
    (sqlite-load-extension db (elisa--vss-path))
    (sqlite-execute db (elisa-embeddings-create-table-sql))
    (sqlite-execute db (elisa-info-create-table-sql))
    (sqlite-execute db (elisa-collections-create-table-sql))
    (sqlite-execute db (elisa-kinds-create-table-sql))
    (sqlite-execute db (elisa-fill-kinds-sql))
    (sqlite-execute db (elisa-files-create-table-sql))
    (sqlite-execute db (elisa-data-create-table-sql))
    (sqlite-execute db (elisa-data-embeddings-create-table-sql))
    (sqlite-execute db (elisa-data-fts-create-table-sql))))

(defvar elisa-db
  (let ((_ (make-directory elisa-db-directory t))
        (db (sqlite-open (file-name-concat elisa-db-directory "elisa.sqlite"))))
    (elisa--init-db db)
    db))

(defun elisa-vector-to-sqlite (data)
  "Convert DATA to sqlite vector representation."
  (format "vector_from_json(json('%s'))" (json-encode data)))

(defun elisa-sqlite-escape (string)
  "Escape single quotes in STRING for sqlite."
  (let ((reps '(("'" . "''")
                ("\\" . "\\\\")
                ("\0" . "\n"))))
    (replace-regexp-in-string
     (regexp-opt (mapcar #'car reps))
     (lambda (str) (alist-get str reps nil nil #'string=))
     string nil t)))

(defun elisa-sqlite-format-int-list (ids)
  "Convert list of integer IDS list to sqlite list representation."
  (format
   "(%s)"
   (mapconcat (lambda (id) (format "%d" id)) ids ", ")))

(defun elisa-sqlite-format-string-list (names)
  "Convert list of string NAMES list to sqlite list representation."
  (format
   "(%s)"
   (mapconcat (lambda (name)
		(format "'%s'"
			(elisa-sqlite-escape name)))
              names ", ")))

(defun elisa-avg (list)
  "Calculate arithmetic average value of LIST."
  (cl-loop for elem in list for count from 0
           summing elem into sum
           finally (return (/ sum (float count)))))

(defun elisa-std-dev (lst)
  "Calculate standart deviation value of LST."
  (let ((avg (elisa-avg lst))
	(len (length lst)))
    (sqrt (/ (cl-reduce
	      #'+
	      (mapcar
	       (lambda (x) (expt (- x avg) 2))
	       lst))
	     len))))

(defun elisa-calculate-threshold (k distances)
  "Calculate breakpoint threshold for DISTANCES based on K standard deviations."
  (+ (elisa-avg distances) (* k (elisa-std-dev distances))))

(defun elisa-string-empty-p (s)
  "Check if string S contain only spacing."
  (length= (string-trim s) 0))

(defun elisa-filter-strings (chunks)
  "Filter out empty CHUNKS."
  (cl-remove-if #'elisa-string-empty-p chunks))

(defun elisa-embeddings (chunks)
  "Calculate embeddings for CHUNKS.
Return list of vectors."
  (let ((provider elisa-embeddings-provider))
    (if (and elisa-batch-embeddings-enabled
	     (member 'embeddings-batch (llm-capabilities provider)))
	(let ((batches (seq-partition chunks elisa-batch-size)))
	  (flatten-list (mapcar (lambda (batch) (llm-batch-embeddings provider (vconcat batch)))
				batches)))
      (mapcar (lambda (chunk) (llm-embedding provider chunk)) chunks))))

(defun elisa-parse-info-manual (name collection-name)
  "Parse info manual with NAME and save index to COLLECTION-NAME."
  (with-temp-buffer
    (ignore-errors
      (info name (current-buffer))
      (let ((collection-id (or (caar (sqlite-select
				      elisa-db
				      (format
				       "SELECT rowid FROM collections WHERE name = '%s';"
				       collection-name)))
			       (progn
				 (sqlite-execute
				  elisa-db
				  (format
				   "INSERT INTO collections (name) VALUES ('%s');"
				   collection-name))
				 (caar (sqlite-select
					elisa-db
					(format
					 "SELECT rowid FROM collections WHERE name = '%s';"
					 collection-name))))))
	    (kind-id (caar (sqlite-select
			    elisa-db "SELECT rowid FROM kinds WHERE name = 'info';")))
	    (continue t)
	    (parsed-nodes nil))
	(while continue
	  (let* ((node-name (concat "(" (file-name-sans-extension
					 (file-name-nondirectory Info-current-file))
				    ") "
				    Info-current-node))
		 (chunks (elisa-split-semantically)))
	    (if (not (cl-find node-name parsed-nodes :test 'string-equal))
		(progn
		  (mapc
		   (lambda (text)
		     (let* ((hash (secure-hash 'sha256 text))
			    (embedding (llm-embedding elisa-embeddings-provider text))
			    (rowid
			     (if-let ((rowid (caar (sqlite-select
						    elisa-db
						    (format "SELECT rowid FROM data WHERE kind_id = %s AND collection_id = %s AND path = '%s' AND hash = '%s';"
							    kind-id collection-id
							    (elisa-sqlite-escape node-name) hash)))))
				 nil
			       (sqlite-execute
				elisa-db
				(format
				 "INSERT INTO data(kind_id, collection_id, path, hash, data) VALUES (%s, %s, '%s', '%s', '%s');"
				 kind-id collection-id
				 (elisa-sqlite-escape node-name) hash (elisa-sqlite-escape text)))
			       (caar (sqlite-select
				      elisa-db
				      (format "SELECT rowid FROM data WHERE kind_id = %s AND collection_id = %s AND path = '%s' AND hash = '%s';"
					      kind-id collection-id
					      (elisa-sqlite-escape node-name) hash))))))
		       (when rowid
			 (sqlite-execute
			  elisa-db
			  (format "INSERT INTO data_embeddings(rowid, embedding) VALUES (%s, %s);"
				  rowid (elisa-vector-to-sqlite embedding)))
			 (sqlite-execute
			  elisa-db
			  (format "INSERT INTO data_fts(rowid, data) VALUES (%s, '%s');"
				  rowid (elisa-sqlite-escape text))))))
		   chunks)
		  (push node-name parsed-nodes)
		  (condition-case nil
		      (funcall-interactively #'Info-forward-node)
		    (error
		     (setq continue nil))))
	      (setq continue nil))))))))

(defun elisa--find-similar (text collections)
  "Find similar to TEXT results in COLLECTIONS.
Return sqlite query.  For asyncronous execution."
  (let* ((rowids (flatten-tree
		  (sqlite-select
		   elisa-db
		   (format "SELECT rowid FROM data WHERE collection_id IN
 (
SELECT rowid FROM collections WHERE name IN %s
);"
			   (elisa-sqlite-format-string-list collections)))))
	 (query (format "WITH
vector_search AS (
  SELECT rowid, distance
  FROM data_embeddings
  WHERE vss_search(embedding, %s)
  ORDER BY distance ASC
  LIMIT 40
),
semantic_search AS (
  SELECT rowid, RANK () OVER (ORDER BY distance ASC) AS rank
  FROM vector_search
  WHERE rowid IN %s
  ORDER BY distance ASC
  LIMIT 20
),
keyword_search AS (
  SELECT rowid, RANK () OVER (ORDER BY bm25(data_fts) ASC) AS rank
  FROM data_fts
  WHERE rowid in %s and data_fts MATCH '%s'
  ORDER BY bm25(data_fts) ASC
  LIMIT 20
),
hybrid_search AS (
SELECT
  COALESCE(semantic_search.rowid, keyword_search.rowid) AS rowid,
  COALESCE(1.0 / (60 + semantic_search.rank), 0.0) +
  COALESCE(1.0 / (60 + keyword_search.rank), 0.0) AS score
FROM semantic_search
FULL OUTER JOIN keyword_search ON semantic_search.rowid = keyword_search.rowid
ORDER BY score DESC
LIMIT %d
)
SELECT
  hybrid_search.rowid
FROM hybrid_search
;
"
			(elisa-vector-to-sqlite
			 (llm-embedding elisa-embeddings-provider text))
			(elisa-sqlite-format-int-list rowids)
			(elisa-sqlite-format-int-list rowids)
			(elisa-fts-query text)
			(elisa-get-limit))))
    query))

(defun elisa-find-similar (text collections on-done)
  "Find similar to TEXT results in COLLECTIONS.
Evaluate ON-DONE with result."
  (message "searching in collected data")
  (elisa--async-do
   (lambda () (elisa--find-similar text collections))
   on-done))

(defun elisa--split-by (func)
  "Split buffer content to list by FUNC."
  (let ((pt (point-min))
	(result nil))
    (save-excursion
      (goto-char (point-min))
      (while (not (eobp))
	(funcall func)
	(push (buffer-substring-no-properties pt (point)) result)
	(setq pt (point)))
      (nreverse (cl-remove-if #'string-empty-p result)))))

(defun elisa-split-by-sentence ()
  "Split byffer to list of sentences."
  (elisa--split-by #'forward-sentence))

(defun elisa-split-by-paragraph ()
  "Split buffer to list of paragraphs."
  (elisa--split-by #'forward-paragraph))

(defun elisa-dot-product (v1 v2)
  "Calculate the dot produce of vectors V1 and V2."
  (let ((result 0))
    (dotimes (i (length v1))
      (setq result (+ result (* (aref v1 i) (aref v2 i)))))
    result))

(defun elisa-magnitude (v)
  "Calculate magnitude of vector V."
  (let ((sum 0))
    (dotimes (i (length v))
      (setq sum (+ sum (* (aref v i) (aref v i)))))
    (sqrt sum)))

(defun elisa-cosine-similarity (v1 v2)
  "Calculate the cosine similarity of V1 and V2.
The return is a floating point number between 0 and 1, where the
closer it is to 1, the more similar it is."
  (let ((dot-product (elisa-dot-product v1 v2))
        (v1-magnitude (elisa-magnitude v1))
        (v2-magnitude (elisa-magnitude v2)))
    (if (and v1-magnitude v2-magnitude)
        (/ dot-product (* v1-magnitude v2-magnitude))
      0)))

(defun elisa-cosine-distance (v1 v2)
  "Calculate cosine-distance between V1 and V2."
  (- 1 (elisa-cosine-similarity v1 v2)))

(defun elisa--similarities (list)
  "Calculate cosine similarities between neighbour elements in LIST."
  (let ((head (car list))
	(tail (cdr list))
	(result nil))
    (while tail
      (push (elisa-cosine-similarity head (car tail)) result)
      (setq head (car tail))
      (setq tail (cdr tail)))
    (nreverse result)))

(defun elisa--distances (list)
  "Calculate cosine distances between neighbour elements in LIST."
  (let ((head (car list))
	(tail (cdr list))
	(result nil))
    (while tail
      (push (elisa-cosine-distance head (car tail)) result)
      (setq head (car tail))
      (setq tail (cdr tail)))
    (nreverse result)))

(defun elisa-split-semantically (&rest args)
  "Split buffer data semantically.
ARGS contains keys for fine control.

:function FUNC -- FUNC is a function for split buffer into chunks.

:threshold-amount K -- K is a breakpoint threshold amount.

than T, it will be packed into single semantic chunk."
  (if-let* ((func (or (plist-get args :function) elisa-semantic-split-function))
	    (k (or (plist-get args :threshold-amount) elisa-breakpoint-threshold-amount))
	    (chunks (elisa-filter-strings (funcall func)))
	    (embeddings (elisa-embeddings chunks))
	    (distances (elisa--distances embeddings))
	    (threshold (elisa-calculate-threshold k distances))
	    (current (car chunks))
	    (tail (cdr chunks)))
      (let* ((result nil))
        (dolist (el distances)
          (if (<= el threshold)
	      (setq current (concat current (car tail)))
	    (push current result)
	    (setq current (car tail)))
	  (setq tail (cdr tail)))
	(push current result)
	(cl-remove-if
	 #'string-empty-p
	 (mapcar (lambda (s)
		   (if s
		       (string-trim s)
		     ""))
		 (nreverse result))))
    (list (buffer-substring-no-properties (point-min) (point-max)))))

(defun elisa--read-ignore-file-regexps (directory)
  "Read ignore patterns from `elisa-ignore-patterns-files' in DIRECTORY."
  (mapcar #'wildcard-to-regexp
	  (flatten-tree
	   (mapcar (lambda (file)
		     (let ((filepath (expand-file-name file directory)))
		       (when (file-exists-p filepath)
			 (with-temp-buffer
			   (insert-file-contents filepath)
			   (split-string (buffer-string) "\n" t)))))
		   elisa-ignore-patterns-files))))

(defun elisa--text-file-p (filename)
  "Check if FILENAME contain text."
  (or (and (get-file-buffer filename) t) ;; if file opened assume it text
      (with-current-buffer (find-file-noselect filename t t)
	(prog1
	    ;; if there is null byte in file, file is binary
	    (not (search-forward "\0" nil t 1))
	  (kill-buffer)))))

(defun elisa--file-list (directory)
  "List of files to parse in DIRECTORY."
  (let ((ignore-regexps (elisa--read-ignore-file-regexps directory)))
    (when elisa-ignore-invisible-files
      (push "$\\.[^/]*" ignore-regexps)
      (push "/\\.[^/]*" ignore-regexps))
    (seq-filter (lambda (file)
		  (and (not (seq-some (lambda (regexp)
					(string-match-p regexp file))
				      ignore-regexps))
		       (or
			(elisa-supported-complex-document-p file)
			(elisa--text-file-p file))))
		(directory-files-recursively directory ".*"))))

(defun elisa-parse-file (collection-id path &optional force)
  "Parse file PATH for COLLECTION-ID.
When FORCE parse even if already parsed."
  (let* ((opened (get-file-buffer path))
	 (buf (if (elisa-supported-complex-document-p path)
		  (funcall elisa-complex-file-extraction-function path)
		(or opened (find-file-noselect path t t))))
	 (hash (secure-hash 'sha256 buf))
	 (prev-hash (caar (sqlite-select
			   elisa-db
			   (format "SELECT hash FROM files WHERE path = '%s';"
				   (elisa-sqlite-escape path))))))
    (when (or force
	      (not prev-hash)
	      (not (string-equal hash prev-hash)))
      (with-current-buffer buf
	(let ((chunks (elisa-split-semantically))
	      (old-row-ids
	       (flatten-tree (sqlite-select
			      elisa-db
			      (format "SELECT rowid FROM data WHERE path = '%s';"
				      (elisa-sqlite-escape path)))))
	      (row-ids nil)
	      (kind-id (caar (sqlite-select
			      elisa-db
			      "SELECT rowid FROM kinds WHERE name = 'file';"))))
	  ;; remove old data
	  (when prev-hash
	    (sqlite-execute
	     elisa-db
	     (format "DELETE FROM files WHERE path = '%s';"
		     (elisa-sqlite-escape path))))
	  ;; add new data
          (dolist (text chunks)
            (let* ((hash (secure-hash 'sha256 text))
		   (rowid
		    (if-let ((rowid (caar (sqlite-select
					   elisa-db
					   (format "SELECT rowid FROM data WHERE kind_id = %s AND collection_id = %s AND path = '%s' AND hash = '%s';"
						   kind-id collection-id
						   (elisa-sqlite-escape path) hash)))))
			(progn
			  (push rowid row-ids)
			  nil)
		      (sqlite-execute
		       elisa-db
		       (format
			"INSERT INTO data(kind_id, collection_id, path, hash, data) VALUES (%s, %s, '%s', '%s', '%s');"
			kind-id collection-id
			(elisa-sqlite-escape path) hash (elisa-sqlite-escape text)))
		      (caar (sqlite-select
			     elisa-db
			     (format "SELECT rowid FROM data WHERE kind_id = %s AND collection_id = %s AND path = '%s' AND hash = '%s';"
				     kind-id collection-id
				     (elisa-sqlite-escape path) hash))))))
	      (when rowid
		(sqlite-execute
		 elisa-db
		 (format "INSERT INTO data_embeddings(rowid, embedding) VALUES (%s, %s);"
			 rowid (elisa-vector-to-sqlite
				(llm-embedding elisa-embeddings-provider text))))
		(sqlite-execute
		 elisa-db
		 (format "INSERT INTO data_fts(rowid, data) VALUES (%s, '%s');"
			 rowid (elisa-sqlite-escape text)))
		(push rowid row-ids))))
	  ;; remove old data
	  (when row-ids
	    (let ((delete-rows (cl-remove-if (lambda (id)
					       (cl-find id row-ids))
					     old-row-ids)))
	      (elisa--delete-data delete-rows)))
	  ;; save hash to files table
	  (sqlite-execute
	   elisa-db
	   (format "INSERT INTO files (path, hash) VALUES ('%s', '%s');"
		   (elisa-sqlite-escape path) hash)))))
    ;; kill buffer if it was not open before parsing
    (when (not opened)
      (kill-buffer buf))))

(defun elisa--delete-from-table (table ids)
  "Delete IDS from TABLE."
  (sqlite-execute
   elisa-db
   (format "DELETE FROM %s WHERE rowid IN %s;"
	   table
	   (elisa-sqlite-format-int-list ids))))

(defun elisa--delete-data (ids)
  "Delete data with IDS."
  (elisa--delete-from-table "data_fts" ids)
  (elisa--delete-from-table "data_embeddings" ids)
  (elisa--delete-from-table "data" ids))

(defun elisa-parse-directory (dir)
  "Parse DIR as new collection syncronously."
  (setq dir (expand-file-name dir))
  (let* ((collection-id (progn
			  (sqlite-execute
			   elisa-db
			   (format
			    "INSERT INTO collections (name) VALUES ('%s') ON CONFLICT DO NOTHING;"
			    (elisa-sqlite-escape dir)))
			  (caar (sqlite-select
				 elisa-db
				 (format
				  "SELECT rowid FROM collections WHERE name = '%s';"
				  (elisa-sqlite-escape dir))))))
	 (files (elisa--file-list dir))
	 (delete-ids (flatten-tree
		      (sqlite-select
		       elisa-db
		       (format
			"SELECT rowid FROM data WHERE collection_id = %d AND path NOT IN %s;"
			collection-id
			(elisa-sqlite-format-string-list files))))))
    (elisa--delete-data delete-ids)
    (dolist (file files)
      (message "parsing %s" file)
      (elisa-parse-file collection-id file))))

;;;###autoload
(defun elisa-async-parse-directory (dir)
  "Parse DIR as new collection asyncronously."
  (interactive "DSelect directory: ")
  (elisa--async-do (lambda ()
		     (elisa-parse-directory
		      (expand-file-name dir)))))

(defvar eww-accept-content-types)

(defun elisa-search-duckduckgo (prompt)
  "Search duckduckgo for PROMPT and return list of urls."
  (require 'eww)
  (let* ((url (format "https://duckduckgo.com/html/?q=%s" (url-hexify-string prompt)))
	 (buffer-name (plz 'get url :as 'buffer
			:headers `(("Accept" . ,eww-accept-content-types)
				   ("Accept-Encoding" . "gzip")
				   ("User-Agent" . ,(url-http--user-agent-default-string))))))
    (with-current-buffer buffer-name
      (goto-char (point-min))
      (search-forward "<!DOCTYPE")
      (beginning-of-line)
      (cl-remove-if
       #'string-empty-p
       (cl-remove-duplicates
	(mapcar
	 (lambda (el)
	   (when el
	     (string-trim-right
	      (url-unhex-string
	       (cdar (url-parse-args (or (dom-attr el 'href) ""))))
	      "[&\\?].*")))
	 (dom-by-tag
	  (libxml-parse-html-region
	   (point) (point-max))
	  'a))
	:test #'string-equal)))))

(defun elisa-starts-with-lowercase-p (string)
  "Check if STRING start with lowercase character."
  (let ((category (get-char-code-property (seq-first string) 'general-category)))
    (or (eq 'Ll category)
	(eq 'Ps category))))

(defun elisa-dehyphen (text)
  "Dehyphen TEXT."
  (ignore-errors (with-temp-buffer
		   (insert (string-join
			    (mapcar #'string-trim (string-split text "\n"))
			    "\n"))
		   (goto-char (point-min))
		   (while (not (eobp))
		     (end-of-line)
		     (if (eq (preceding-char) ?-)
			 (progn
			   (delete-char 1)
			   (delete-char -1))
		       (forward-line)))
		   (buffer-substring-no-properties (point-min) (point-max)))))

(defun elisa-parse-with-tika-buffer (file)
  "Parse FILE with tika."
  (let* ((url (format "%s/tika" (string-trim-right elisa-tika-url "/")))
	 (buf (plz 'put url :body (list 'file file) :as 'buffer))
	 (shr-use-fonts nil)
	 (shr-width (- ellama-long-lines-length 5))
	 (data (with-current-buffer buf
		 (libxml-parse-html-region (point-min) (point-max))))
	 (prev-elt nil))
    (dolist (elt (dom-by-tag data 'p))
      (dolist (text (dom-children elt))
	;; trim string content
	(when-let* ((trimmed-text (string-trim text))
		    (new-elt (if (or (string-match "^[0-9]+$" trimmed-text)
				     (string= "" trimmed-text))
				 (progn (dom-remove-node data elt)
					nil)
			       (if (elisa-starts-with-lowercase-p trimmed-text)
				   (progn
				     (dom-remove-node data prev-elt)
				     (dom-node 'p nil (elisa-dehyphen
						       (concat
							(car (dom-children prev-elt))
							"\n" trimmed-text))))
				 (dom-node 'p nil (elisa-dehyphen trimmed-text))))))
	  (setq prev-elt new-elt)
	  (setq data (cl-nsubst new-elt elt data :test #'equal))))
      (when (eq (length (dom-children elt)) 0)
	(dom-remove-node data elt)))
    (with-current-buffer buf
      (delete-region (point-min) (point-max))
      (ignore-errors
	(shr-insert-document data))
      buf)))

(defun elisa-search-searxng (prompt)
  "Search searxng for PROMPT and return list of urls.
You can customize `elisa-searxng-url' to use non local instance."
  (let ((url (format "%s/search?format=json&q=%s" elisa-searxng-url (url-hexify-string prompt))))
    (thread-last
      (plz 'get url :as #'json-read)
      (alist-get 'results)
      (mapcar (lambda (el) (alist-get 'url el))))))

(defun elisa-get-webpage-buffer (url)
  "Get buffer with URL content."
  (require 'eww)
  (let ((buffer-name (ignore-errors
		       (plz 'get url :as 'buffer
			 :headers `(("Accept" . ,eww-accept-content-types)
				    ("Accept-Encoding" . "gzip")
				    ("User-Agent" . ,(url-http--user-agent-default-string))))))
	;; fix one word lines for async execution
	(shr-use-fonts nil)
	(shr-width (- ellama-long-lines-length 5)))
    (when buffer-name
      (with-current-buffer buffer-name
	(goto-char (point-min))
	(or (search-forward "<!DOCTYPE" nil t)
            (search-forward "<html" nil t))
	(beginning-of-line)
	(kill-region (point-min) (point))
	(ignore-errors
	  (shr-insert-document (libxml-parse-html-region (point-min) (point-max))))
	(goto-char (point-min))
	(or (search-forward "<!DOCTYPE" nil t)
            (search-forward "<html" nil t))
	(beginning-of-line)
	(kill-region (point) (point-max))
	buffer-name))))

(defun elisa-get-webpage-buffer-pandoc (url)
  "Get buffer with URL content translated to markdown with pandoc."
  (let ((buffer-name (plz 'get url :as 'buffer)))
    (with-current-buffer buffer-name
      (shell-command-on-region
       (point-min) (point-max)
       (format "%s --from html --to plain" elisa-pandoc-executable)
       buffer-name t)
      buffer-name)))

(defun elisa-fts-query (prompt)
  "Return fts match query for PROMPT."
  (thread-last
    prompt
    (string-trim)
    (downcase)
    (string-replace "-" " ")
    (replace-regexp-in-string "[^[:alnum:] ]+" "")
    (string-trim)
    (replace-regexp-in-string "[[:space:]]+" " OR ")))

(defun elisa--rerank-request (prompt ids)
  "Generate rerank request body for PROMPT and IDS."
  (let ((docs
	 (mapcar
	  (lambda (row)
	    (let ((id (cl-first row))
		  (text (cl-second row)))
	      `(("id" . ,id) ("text" . ,text))))
	  (sqlite-select
	   elisa-db
	   (format
	    "SELECT rowid, data FROM data WHERE rowid IN %s;"
	    (elisa-sqlite-format-int-list ids))))))
    (json-encode `(("query" . ,prompt)
		   ("documents" . ,docs)))))

(defun elisa--do-rerank-request (prompt ids)
  "Call rerank service for PROMPT and IDS."
  (when ids
    (seq--into-list
     (alist-get 'data
		(plz 'post (format "%s/api/v1/rerank"
				   (string-remove-suffix "/" elisa-reranker-url))
		  :headers `(("Content-Type" . "application/json"))
		  :body-type 'text
		  :body (elisa--rerank-request prompt ids)
		  :as #'json-read)))))

(defun elisa-rerank (prompt ids)
  "Rerank IDS according to PROMPT and return top `elisa-limit' IDS."
  (let ((data (elisa--do-rerank-request prompt ids)))
    (mapcar (lambda (elt)
	      (alist-get 'id elt))
	    (take elisa-limit
		  (if elisa-reranker-similarity-threshold
		      (cl-remove-if (lambda (obj)
				      (< (alist-get 'similarity obj)
					 elisa-reranker-similarity-threshold))
				    data)
		    data)))))

(defun elisa-get-limit ()
  "Limit for elisa hybrid search."
  (if elisa-reranker-enabled
      elisa-reranker-limit
    elisa-limit))

(defun elisa--parse-web-page (collection-id url)
  "Parse URL into collection with COLLECTION-ID."
  (let ((kind-id (caar (sqlite-select
			elisa-db "SELECT rowid FROM kinds WHERE name = 'web';"))))
    (message "collecting data from %S..." url)
    (dolist (chunk (elisa-extact-webpage-chunks url))
      (let* ((hash (secure-hash 'sha256 chunk))
	      (embedding (llm-embedding elisa-embeddings-provider chunk))
	      (rowid
	       (if-let ((rowid (caar (sqlite-select
				      elisa-db
				      (format "SELECT rowid FROM data WHERE kind_id = %s AND collection_id = %s AND path = '%s' AND hash = '%s';" kind-id collection-id url hash)))))
		   nil
		 (sqlite-execute
		  elisa-db
		  (format
		   "INSERT INTO data(kind_id, collection_id, path, hash, data) VALUES (%s, %s, '%s', '%s', '%s');"
		   kind-id collection-id url hash (elisa-sqlite-escape chunk)))
		 (caar (sqlite-select
			elisa-db
			(format "SELECT rowid FROM data WHERE kind_id = %s AND collection_id = %s AND path = '%s' AND hash = '%s';" kind-id collection-id url hash))))))
	 (when rowid
	   (sqlite-execute
	    elisa-db
	    (format "INSERT INTO data_embeddings(rowid, embedding) VALUES (%s, %s);"
		    rowid (elisa-vector-to-sqlite embedding)))
	   (sqlite-execute
	    elisa-db
	    (format "INSERT INTO data_fts(rowid, data) VALUES (%s, '%s');"
		    rowid (elisa-sqlite-escape chunk))))))))

(defun elisa--web-search (prompt)
  "Search the web for PROMPT.
Return sqlite query that extract data for adding to context."
  (sqlite-execute
   elisa-db
   (format
    "INSERT INTO collections (name) VALUES ('%s') ON CONFLICT DO NOTHING;"
    (elisa-sqlite-escape prompt)))
  (let* ((collection-id (caar (sqlite-select
			       elisa-db
			       (format
				"SELECT rowid FROM collections WHERE name = '%s';"
				(elisa-sqlite-escape prompt)))))
	 (urls (funcall elisa-web-search-function prompt))
	 (collected-pages 0))
    (dolist (url urls)
      (when (<= collected-pages elisa-web-pages-limit)
	(elisa--parse-web-page collection-id url)
	(cl-incf collected-pages)))))

(defun elisa--rewrite-prompt (prompt action)
  "Rewrite PROMPT if `elisa-prompt-rewriting-enabled'.
Call ACTION with new prompt."
  (let ((session (and ellama--current-session-id
		      (with-current-buffer (ellama-get-session-buffer
					    ellama--current-session-id)
			ellama--current-session))))
    (if (and elisa-prompt-rewriting-enabled
	     ellama--current-session-id
	     (string= (llm-name (ellama-session-provider session))
		      (llm-name elisa-chat-provider)))
	(with-current-buffer (get-buffer-create (make-temp-name "elisa"))
	  (ellama-stream
	   (format elisa-rewrite-prompt-template prompt)
	   :session session
	   :buffer (current-buffer)
	   :provider elisa-chat-provider
	   :on-done action))
      (funcall action prompt))))

;;;###autoload
(defun elisa-web-search (prompt)
  "Search the web for PROMPT."
  (interactive "sAsk elisa with web search: ")
  (elisa--rewrite-prompt prompt #'elisa--web-search-internal))

(defun elisa--web-search-internal (prompt)
  "Search the web for PROMPT."
  (message "searching the web")
  (elisa--async-do
   (lambda () (elisa--web-search prompt))
   (lambda (_)
     (elisa-find-similar
      prompt (list prompt)
      (lambda (query) (elisa-retrieve-ask query prompt))))))

(defun elisa-retrieve-ask (query prompt)
  "Retrieve data with QUERY and ask elisa for PROMPT."
  (elisa--async-do
   (lambda () (let* ((raw-ids (flatten-tree (sqlite-select elisa-db query)))
		     (ids (if elisa-reranker-enabled
			      (elisa-rerank prompt raw-ids)
			    (take elisa-limit raw-ids))))
		(when ids
		  (sqlite-select
		   elisa-db
		   (format
		    "SELECT k.name, d.path, d.data
FROM data AS d
LEFT JOIN kinds k ON k.rowid = d.kind_id
WHERE d.rowid in %s;"
		    (elisa-sqlite-format-int-list ids))))))
   (lambda (result)
     (if result (mapc
		 (lambda (row)
		   (when-let ((kind (cl-first row))
			      (path (cl-second row))
			      (text (cl-third row)))
		     (pcase kind
		       ("web"
			(ellama-context-add-webpage-quote-noninteractive path path text))
		       ("file"
			(ellama-context-add-file-quote-noninteractive path text))
		       ("info"
			(ellama-context-add-info-node-quote-noninteractive path text)))))
		 result)
       (ellama-context-add-text "No related documents found."))
     (ellama-chat
      (format elisa-chat-prompt-template prompt)
      nil :provider elisa-chat-provider))))

(defun elisa--info-valid-p (name)
  "Return NAME if info is valid."
  (with-temp-buffer
    (ignore-errors
      (info name (current-buffer))
      name)))

(defun elisa-get-builtin-manuals ()
  "Get builtin manual names list."
  (mapcar
   #'file-name-base
   (cl-remove-if-not
    (lambda (s)
      (or (string-suffix-p ".info" s)
	  (string-suffix-p ".info.gz" s)))
    (directory-files (with-temp-buffer
		       (info "emacs" (current-buffer))
		       (file-name-directory Info-current-file))))))

(defun elisa-get-external-manuals ()
  "Get external manual names list."
  (thread-last
    (process-lines
     elisa-find-executable
     (file-truename (file-name-concat user-emacs-directory "elpa"))
     "-name" "*.info")
    (mapcar #'file-name-base)
    (seq-uniq)
    (mapcar #'elisa--info-valid-p)
    (cl-remove-if #'not)))

(defun elisa-parse-builtin-manuals ()
  "Parse builtin manuals."
  (mapc (lambda (s)
	  (elisa-parse-info-manual s "builtin manuals"))
	(elisa-get-builtin-manuals)))

(defun elisa-parse-external-manuals ()
  "Parse external manuals."
  (mapc (lambda (s)
	  (elisa-parse-info-manual s "external manuals"))
	(elisa-get-external-manuals)))

(defun elisa-parse-all-manuals ()
  "Parse all manuals."
  (elisa-parse-builtin-manuals)
  (elisa-parse-external-manuals))

(defun elisa--reopen-db ()
  "Reopen database."
  (let ((db (sqlite-open (file-name-concat elisa-db-directory "elisa.sqlite"))))
    (elisa--init-db db)
    (setq elisa-db db)))

(defun elisa--async-do (func &optional on-done)
  "Do FUNC asyncronously.
Call ON-DONE callback with result as an argument after FUNC evaluation done."
  (let* ((command real-this-command)
	 (reporter (make-progress-reporter (if command
					       (prin1-to-string command)
					     "elisa async processing")))
	 (timer (run-at-time t 0.2 (lambda () (progress-reporter-update reporter)))))
    (async-start `(lambda ()
		    ,(async-inject-variables "elisa-embeddings-provider")
		    ,(async-inject-variables "elisa-db-directory")
		    ,(async-inject-variables "elisa-find-executable")
		    ,(async-inject-variables "elisa-tar-executable")
		    ,(async-inject-variables "elisa-prompt-rewriting-enabled")
		    ,(async-inject-variables "elisa-batch-embeddings-enabled")
		    ,(async-inject-variables "elisa-batch-size")
		    ,(async-inject-variables "elisa-rewrite-prompt-template")
		    ,(async-inject-variables "elisa-semantic-split-function")
		    ,(async-inject-variables "elisa-webpage-extraction-function")
		    ,(async-inject-variables "elisa-supported-complex-document-extensions")
		    ,(async-inject-variables "elisa-complex-file-extraction-function")
		    ,(async-inject-variables "elisa-web-search-function")
		    ,(async-inject-variables "elisa-tika-url")
		    ,(async-inject-variables "elisa-searxng-url")
		    ,(async-inject-variables "elisa-web-pages-limit")
		    ,(async-inject-variables "elisa-breakpoint-threshold-amount")
		    ,(async-inject-variables "elisa-pandoc-executable")
		    ,(async-inject-variables "ellama-long-lines-length")
		    ,(async-inject-variables "elisa-reranker-enabled")
		    ,(async-inject-variables "elisa-sqlite-vector-path")
		    ,(async-inject-variables "elisa-sqlite-vss-path")
		    ,(async-inject-variables "load-path")
		    ,(async-inject-variables "Info-directory-list")
		    (require 'elisa)
		    (,func))
		 (lambda (res)
		   (cancel-timer timer)
		   (progress-reporter-done reporter)
		   (sqlite-close elisa-db)
		   (elisa--reopen-db)
		   (when on-done
		     (funcall on-done res))))))

(defun elisa-extact-webpage-chunks (url)
  "Extract semantic chunks for webpage fetched from URL."
  (when-let ((buf (funcall elisa-webpage-extraction-function url)))
    (with-current-buffer buf
      (elisa-split-semantically))))

;;;###autoload
(defun elisa-async-parse-builtin-manuals ()
  "Parse builtin manuals asyncronously."
  (interactive)
  (message "Begin parsing builtin manuals.")
  (elisa--async-do 'elisa-parse-builtin-manuals))

;;;###autoload
(defun elisa-async-parse-external-manuals ()
  "Parse external manuals asyncronously."
  (interactive)
  (message "Begin parsing external manuals.")
  (elisa--async-do 'elisa-parse-external-manuals))

;;;###autoload
(defun elisa-async-parse-all-manuals ()
  "Parse all manuals asyncronously."
  (interactive)
  (message "Begin parsing manuals.")
  (elisa--async-do 'elisa-parse-all-manuals))

;;;###autoload
(defun elisa-reparse-current-collection ()
  "Incrementally reparse current directory collection.
It does nothing if buffer file not inside one of existing collections."
  (interactive)
  (when-let* ((collections (flatten-tree
			    (sqlite-select
			     elisa-db
			     "SELECT name FROM collections;")))
	      (dirs (cl-remove-if-not #'file-directory-p collections))
	      (file (buffer-file-name))
	      (collection (cl-find-if (lambda (dir)
					(file-in-directory-p file dir))
				      dirs)))
    (elisa-async-parse-directory collection)))

;;;###autoload
(defun elisa-disable-collection (&optional collection)
  "Disable COLLECTION."
  (interactive)
  (let ((col (or collection
		 (completing-read
		  "Disable collection: "
		  elisa-enabled-collections))))
    (setq elisa-enabled-collections
	  (cl-remove col elisa-enabled-collections :test #'string=))))

;;;###autoload
(defun elisa-disable-all-collections ()
  "Disable all collections."
  (interactive)
  (mapc #'elisa-disable-collection elisa-enabled-collections))

;;;###autoload
(defun elisa-enable-collection (&optional collection)
  "Enable COLLECTION."
  (interactive)
  (let ((col (or collection
		 (completing-read
		  "Enable collection: "
		  (cl-remove-if
		   (lambda (c)
		     (cl-find c elisa-enabled-collections :test #'string=))
		   (flatten-tree
		    (sqlite-select
		     elisa-db
		     "SELECT name FROM collections;")))))))
    (push col elisa-enabled-collections)))

;;;###autoload
(defun elisa-enable-all-collections ()
  "Enable all collections."
  (interactive)
  (let ((all-collections
	 (flatten-tree
	  (sqlite-select
	   elisa-db
	   "SELECT DISTINCT name FROM collections;"))))
    (setq elisa-enabled-collections
	  (cl-set-difference all-collections elisa-enabled-collections :test #'string=))
    (mapc #'elisa-enable-collection all-collections)))

;;;###autoload
(defun elisa-create-empty-collection (&optional collection)
  "Create new empty COLLECTION."
  (interactive "sNew collection name: ")
  (save-window-excursion
    (sqlite-execute
     elisa-db
     (format
      "INSERT INTO collections (name) VALUES ('%s') ON CONFLICT DO NOTHING;"
      (elisa-sqlite-escape collection)))))

;;;###autoload
(defun elisa-add-file-to-collection (file collection)
  "Add FILE to COLLECTION."
  (interactive
   (list
    (read-file-name "File: ")
    (completing-read
     "Enable collection: "
     (flatten-tree
      (sqlite-select
       elisa-db
       "SELECT name FROM collections;")))))
  (let ((collection-id (caar (sqlite-select
			      elisa-db
			      (format
			       "SELECT rowid FROM collections WHERE name = '%s';"
			       (elisa-sqlite-escape collection))))))
    (elisa--async-do (lambda () (elisa-parse-file collection-id file)))))

;;;###autoload
(defun elisa-add-webpage-to-collection (url collection)
  "Add webpage by URL to COLLECTION."
  (interactive
   (list
    (if-let ((url (or (thing-at-point 'url)
                      (shr-url-at-point nil))))
        url
      (read-string "Enter URL you want to summarize: "))
    (completing-read
     "Enable collection: "
     (flatten-tree
      (sqlite-select
       elisa-db
       "SELECT name FROM collections;")))))
  (let ((collection-id (caar (sqlite-select
			      elisa-db
			      (format
			       "SELECT rowid FROM collections WHERE name = '%s';"
			       (elisa-sqlite-escape collection))))))
    (elisa--async-do (lambda () (elisa--parse-web-page collection-id url)))))

;;;###autoload
(defun elisa-remove-collection (&optional collection)
  "Remove COLLECTION."
  (interactive)
  (let* ((col (or collection
		  (completing-read
		   "Enable collection: "
		   (flatten-tree
		    (sqlite-select
		     elisa-db
		     "SELECT name FROM collections;")))))
	 (collection-id (caar (sqlite-select
			       elisa-db
			       (format
				"SELECT rowid FROM collections WHERE name = '%s';"
				(elisa-sqlite-escape col)))))
	 (delete-ids (flatten-tree
		      (sqlite-select
		       elisa-db
		       (format
			"SELECT rowid FROM data WHERE collection_id = %d;"
			collection-id)))))
    (elisa-disable-collection col)
    (when (file-directory-p col)
      (let ((files
	     (flatten-tree
	      (sqlite-select
	       elisa-db
	       (format
		"SELECT DISTINCT path FROM data WHERE collection_id = %d;"
		collection-id)))))
	(sqlite-execute
	 elisa-db
	 (format
	  "DELETE FROM files WHERE path IN %s;"
	  (elisa-sqlite-format-string-list files)))))
    (elisa--delete-data delete-ids)
    (sqlite-execute
     elisa-db
     (format
      "DELETE FROM collections WHERE rowid = %d;"
      collection-id))))

(defun elisa--gen-chat (&optional collections)
  "Generate function for chat with elisa based on COLLECTIONS."
  (let ((cols (or collections elisa-enabled-collections)))
    (lambda (prompt)
      (elisa-find-similar
       prompt cols
       (lambda (query) (elisa-retrieve-ask query prompt))))))

;;;###autoload
(defun elisa-chat (prompt &optional collections)
  "Send PROMPT to elisa.
Find similar quotes in COLLECTIONS and add it to context."
  (interactive "sAsk elisa: ")
  (let ((cols (or collections elisa-enabled-collections)))
    (elisa--rewrite-prompt prompt (elisa--gen-chat cols))))

(defun elisa-recalculate-embeddings ()
  "Recalculate and save new embeddings after embedding provider change."
  (sqlite-execute elisa-db "DELETE FROM data WHERE data = '';") ;; remove rows without data
  (let* ((data-rows (sqlite-select elisa-db "SELECT rowid, data FROM data;"))
	 (texts (mapcar #'cadr data-rows))
	 (rowids (mapcar #'car data-rows))
	 (embeddings (elisa-embeddings texts))
	 (len (length rowids))
	 (i 0))
    ;; Recreate embeddings table
    (sqlite-execute elisa-db (elisa-data-embeddings-drop-table-sql))
    (sqlite-execute elisa-db (elisa-data-embeddings-create-table-sql))
    ;; Recalculate embeddings
    (with-sqlite-transaction elisa-db
      (while (< i len)
	(let ((rowid (nth i rowids))
	      (embedding (nth i embeddings)))
	  (sqlite-execute
	   elisa-db
	   (format "INSERT INTO data_embeddings(rowid, embedding) VALUES (%s, %s);"
		   rowid (elisa-vector-to-sqlite embedding)))
	  (setq i (1+ i)))))))

;;;###autoload
(defun elisa-async-recalculate-embeddings ()
  "Recalculate embeddings asynchronously."
  (interactive)
  (elisa--async-do 'elisa-recalculate-embeddings))

(defun elisa-research-extract-topics-async ()
  "Extract topics from current buffer asynchronously."
  (interactive)
  (ellama-extract-list-async
   "topics"
   ;; TODO: save data into ellama session and start main loop
   (lambda (res)
     (message "extracted topics: %s" res))
   (buffer-substring-no-properties (point-min) (point-max))))

(defun elisa-bind-topic-extraction ()
  "Bind topic extraction."
  (local-set-key (kbd "C-c C-c") #'elisa-research-extract-topics-async)
  (message "Press C-c C-c to start research"))

;;;###autoload
(defun elisa-research-generate-topics (theme)
  "Generate topics for research THEME."
  (interactive "sResearch topic: ")
  (ellama-instant (format
		   elisa-research-topics-generator-template
		   theme)
		  :provider elisa-chat-provider
		  :on-done #'elisa-bind-topic-extraction))

(provide 'elisa)
;;; elisa.el ends here.
