;;; elisa.el --- Emacs Lisp Information System Assistant -*- lexical-binding: t -*-

;; Copyright (C) 2024  Free Software Foundation, Inc.

;; Author: Sergey Kostyaev <sskostyaev@gmail.com>
;; URL: http://github.com/s-kostyaev/elisa
;; Keywords: help local tools
;; Package-Requires: ((emacs "29.2") (ellama "0.11.2") (llm "0.9.1") (async "1.9.8") (plz "0.9"))
;; Version: 1.0.0
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
;; ELISA (Emacs Lisp Information System Assistant) is a system designed
;; to provide informative answers to user queries by leveraging a
;; Retrieval Augmented Generation (RAG) approach.
;;

;;; Code:
(require 'ellama)
(require 'llm)
(require 'info)
(require 'async)
(require 'dom)
(require 'shr)
(require 'plz)
(require 'json)

(defgroup elisa nil
  "RAG implementation for `ellama'."
  :group 'tools)

(defcustom elisa-embeddings-provider (progn (require 'llm-ollama)
					    (make-llm-ollama
					     :embedding-model "nomic-embed-text"))
  "Embeddings provider to generate embeddings."
  :group 'elisa
  :type '(sexp :validate 'cl-struct-p))

(defcustom elisa-chat-provider (progn (require 'llm-ollama)
				      (make-llm-ollama
				       :chat-model "sskostyaev/openchat:8k-rag"
				       :embedding-model "nomic-embed-text"))
  "Chat provider."
  :group 'elisa
  :type '(sexp :validate 'cl-struct-p))

(defcustom elisa-db-directory (file-truename
			       (file-name-concat
				user-emacs-directory "elisa"))
  "Directory for elisa database."
  :group 'elisa
  :type 'directory)

(defcustom elisa-limit 5
  "Count quotes to pass into llm context for answer."
  :group 'elisa
  :type 'integer)

(defcustom elisa-find-executable "find"
  "Path to find executable."
  :group 'elisa
  :type 'string)

(defcustom elisa-tar-executable "tar"
  "Path to tar executable."
  :group 'elisa
  :type 'string)

(defcustom elisa-sqlite-vss-version "v0.1.2"
  "Sqlite VSS version."
  :group 'elisa
  :type 'string)

(defcustom elisa-sqlite-vss-path nil
  "Path to sqlite-vss extension."
  :group 'elisa
  :type 'file)

(defcustom elisa-sqlite-vector-path nil
  "Path to sqlite-vector extension."
  :group 'elisa
  :type 'file)

(defcustom elisa-semantic-split-function 'elisa-split-by-paragraph
  "Function for semantic text split."
  :group 'elisa
  :type 'function)

(defcustom elisa-prompt-rewriting-enabled t
  "Enable prompt rewriting for better retrieving."
  :group 'elisa
  :type 'boolean)

(defcustom elisa-chat-prompt-template "Answer user query based on context above. If you can answer it partially do it. Provide list of open questions if any. Say \"not enough data\" if you can't answer user query based on provided context. User query:
%s"
  "Chat prompt template."
  :group 'elisa
  :type 'string)

(defcustom elisa-rewrite-prompt-template
  "You are professional search agent. With given context and user
prompt you need to create new prompt for search. It should be
concise and useful without additional context. Response with
prompt only. You should replace all words like 'this' or 'it' to
its values to make search successful. If user prompt contains
question your prompt should also be in form of question. For
example:

- What is pony?
- Pony is ...
- How to buy it?

How to buy a pony?

 User prompt:
%s"
  "Prompt template for prompt rewriting."
  :group 'elisa
  :type 'string)

(defcustom elisa-searxng-url "http://localhost:8080/"
  "Searxng url for web search.  Json format should be enabled for this instance."
  :group 'elisa
  :type 'string)

(defcustom elisa-pandoc-executable "pandoc"
  "Path to pandoc executable."
  :group 'elisa
  :type 'string)

(defcustom elisa-webpage-extraction-function 'elisa-get-webpage-buffer
  "Function to get buffer with webpage content."
  :group 'elisa
  :type 'function)

(defcustom elisa-web-search-function 'elisa-search-duckduckgo
  "Function to search the web.
Function should get prompt and return list of urls."
  :group 'elisa
  :type 'function)

(defcustom elisa-web-pages-limit 10
  "Limit of web pages to parse during web search."
  :group 'elisa
  :type 'integer)

(defcustom elisa-breakpoint-threshold-amount 0.4
  "Breakpoint threshold amount.
Increase it if you need decrease semantic split granularity."
  :group 'elisa
  :type 'float)

(defcustom elisa-reranker-enabled nil
  "Enable reranker to improve retrieving quality."
  :group 'elisa
  :type 'boolean)

(defcustom elisa-reranker-url "http://127.0.0.1:8787/"
  "Reranker service url."
  :group 'elisa
  :type 'string)

(defcustom elisa-reranker-similarity-threshold 0
  "Reranker similarity threshold.
If set, all quotes with similarity less than threshold will be filtered out."
  :group 'elisa
  :type 'string)

(defcustom elisa-reranker-limit 20
  "Number of quotes for send to reranker."
  :group 'elisa
  :type 'integer)

(defcustom elisa-ignore-patterns-files '(".gitignore" ".ignore" ".rgignore")
  "Files with patterns to ignore during file parsing."
  :group 'elisa
  :type '(list string))

(defcustom elisa-ignore-invisible-files t
  "Ignore invisible files and directories during file parsing."
  :group 'elisa
  :type 'boolean)

(defcustom elisa-enabled-collections '("builtin manuals" "external manuals")
  "Enabled collections for elisa chat."
  :group 'elisa
  :type '(list string))

(defun elisa-sqlite-vss-download-url ()
  "Generate sqlite vss download url based on current system."
  (cond  ((string-equal system-type "darwin")
	  (if (string-prefix-p "aarch64" system-configuration)
	      (format
	       "https://github.com/asg017/sqlite-vss/releases/download/%s/sqlite-vss-%s-loadable-macos-aarch64.tar.gz"
	       elisa-sqlite-vss-version
	       elisa-sqlite-vss-version)
	    (format
	     "https://github.com/asg017/sqlite-vss/releases/download/%s/sqlite-vss-%s-loadable-macos-x86_64.tar.gz"
	     elisa-sqlite-vss-version
	     elisa-sqlite-vss-version)))
	 ((string-equal system-type "gnu/linux")
	  (format
	   "https://github.com/asg017/sqlite-vss/releases/download/%s/sqlite-vss-%s-loadable-linux-x86_64.tar.gz"
	   elisa-sqlite-vss-version
	   elisa-sqlite-vss-version))
	 (t (user-error "Can't determine download url"))))

(defun elisa--vss-path ()
  "Path to vss sqlite extension."
  (or elisa-sqlite-vss-path
      (let* ((ext (if (string-equal system-type "darwin")
		      "dylib"
		    "so"))
	     (file (format "vss0.%s" ext)))
	(file-name-concat elisa-db-directory file))))

(defun elisa--vector-path ()
  "Path to vector sqlite extension."
  (or elisa-sqlite-vector-path
      (let* ((ext (if (string-equal system-type "darwin")
		      "dylib"
		    "so"))
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
  "drop table if exists elisa_embeddings;")

(defun elisa-data-embeddings-create-table-sql ()
  "Generate sql for create data embeddings table."
  (format "create virtual table if not exists data_embeddings using vss0(embedding(%d));"
	  (elisa-get-embedding-size)))

(defun elisa-data-fts-create-table-sql ()
  "Generate sql for create full text search table."
  "create virtual table if not exists data_fts using fts5(data);")

(defun elisa-info-create-table-sql ()
  "Generate sql for create info table."
  "drop table if exists info;")

(defun elisa-collections-create-table-sql ()
  "Generate sql for create collections table."
  "create table if not exists collections (name text unique);")

(defun elisa-kinds-create-table-sql ()
  "Generate sql for create kinds table."
  "create table if not exists kinds (name text unique);")

(defun elisa-fill-kinds-sql ()
  "Generate sql for fill kinds table."
  "insert into kinds (name) values ('web'), ('file'), ('info') on conflict do nothing;")

(defun elisa-files-create-table-sql ()
  "Generate sql for create files table."
  "create table if not exists files (path text unique, hash text)")

(defun elisa-data-create-table-sql ()
  "Generate sql for create data table."
  "create table if not exists data (
kind_id INTEGER,
collection_id INTEGER,
path text,
hash text,
data text,
FOREIGN KEY(kind_id) REFERENCES kinds(rowid),
FOREIGN KEY(collection_id) REFERENCES collections(rowid)
);")

(defun elisa--init-db (db)
  "Initialize elisa DB."
  (if (not (file-exists-p (elisa--vss-path)))
      (warn "Please run M-x `elisa-download-sqlite-vss' to use this package")
    (sqlite-pragma db "PRAGMA journal_mode=WAL;")
    (sqlite-load-extension
     db
     (elisa--vector-path))
    (sqlite-load-extension
     db
     (elisa--vss-path))
    (sqlite-execute db (elisa-embeddings-create-table-sql))
    (sqlite-execute db (elisa-info-create-table-sql))
    (sqlite-execute db (elisa-collections-create-table-sql))
    (sqlite-execute db (elisa-kinds-create-table-sql))
    (sqlite-execute db (elisa-fill-kinds-sql))
    (sqlite-execute db (elisa-files-create-table-sql))
    (sqlite-execute db (elisa-data-create-table-sql))
    (sqlite-execute db (elisa-data-embeddings-create-table-sql))
    (sqlite-execute db (elisa-data-fts-create-table-sql))))

(defvar elisa-db (progn
		   (make-directory elisa-db-directory t)
		   (let ((db (sqlite-open (file-name-concat elisa-db-directory "elisa.sqlite"))))
		     (elisa--init-db db)
		     db)))

(defun elisa-vector-to-sqlite (data)
  "Convert DATA to sqlite vector representation."
  (format "vector_from_json(json('%s'))"
	  (json-encode data)))

(defun elisa-sqlite-escape (s)
  "Escape single quotes in S for sqlite."
  (thread-last
    s
    (string-replace "'" "''")
    (string-replace "\\" "\\\\")
    (string-replace "\0" "\n")))

(defun elisa-sqlite-format-int-list (ids)
  "Convert list of integer IDS list to sqlite list representation."
  (format
   "(%s)"
   (string-join (mapcar (lambda (id) (format "%d" id)) ids) ", ")))

(defun elisa-sqlite-format-string-list (names)
  "Convert list of string NAMES list to sqlite list representation."
  (format
   "(%s)"
   (string-join (mapcar (lambda (name)
			  (format "'%s'"
				  (elisa-sqlite-escape name))) names) ", ")))

(defun elisa-avg (lst)
  "Calculate arithmetic average value of LST."
  (let ((len (length lst))
	(sum (cl-reduce #'+ lst :initial-value 0.0)))
    (/ sum len)))

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

(defun elisa-parse-info-manual (name collection-name)
  "Parse info manual with NAME and save index to COLLECTION-NAME."
  (with-temp-buffer
    (ignore-errors
      (info name (current-buffer))
      (let ((collection-id (or (caar (sqlite-select
				      elisa-db
				      (format
				       "select rowid from collections where name = '%s';"
				       collection-name)))
			       (progn
				 (sqlite-execute
				  elisa-db
				  (format
				   "insert into collections (name) values ('%s');"
				   collection-name))
				 (caar (sqlite-select
					elisa-db
					(format
					 "select rowid from collections where name = '%s';"
					 collection-name))))))
	    (kind-id (caar (sqlite-select
			    elisa-db "select rowid from kinds where name = 'info';")))
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
						    (format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';"
							    kind-id collection-id
							    (elisa-sqlite-escape node-name) hash)))))
				 nil
			       (sqlite-execute
				elisa-db
				(format
				 "insert into data(kind_id, collection_id, path, hash, data) values (%s, %s, '%s', '%s', '%s');"
				 kind-id collection-id
				 (elisa-sqlite-escape node-name) hash (elisa-sqlite-escape text)))
			       (caar (sqlite-select
				      elisa-db
				      (format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';"
					      kind-id collection-id
					      (elisa-sqlite-escape node-name) hash))))))
		       (when rowid
			 (sqlite-execute
			  elisa-db
			  (format "insert into data_embeddings(rowid, embedding) values (%s, %s);"
				  rowid (elisa-vector-to-sqlite embedding)))
			 (sqlite-execute
			  elisa-db
			  (format "insert into data_fts(rowid, data) values (%s, '%s');"
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
		   (format "select rowid from data where collection_id in
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
      (while (< (point) (point-max))
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
	    (chunks (funcall func))
	    (embeddings (cl-remove-if
			 #'not
			 (mapcar (lambda (s)
				   (when (length> (string-trim s) 0)
				     (llm-embedding elisa-embeddings-provider s)))
				 chunks)))
	    (distances (elisa--distances embeddings))
	    (threshold (elisa-calculate-threshold k distances))
	    (current (car chunks))
	    (tail (cdr chunks)))
      (let* ((result nil))
	(mapc
	 (lambda (el)
	   (if (<= el threshold)
	       (setq current (concat current (car tail)))
	     (push current result)
	     (setq current (car tail)))
	   (setq tail (cdr tail)))
	 distances)
	(push current result)
	(cl-remove-if
	 #'string-empty-p
	 (mapcar (lambda (s)
		   (if s
		       (string-trim s)
		     ""))
		 (nreverse result))))
    (list (buffer-substring-no-properties (point-min) (point-max)))))

(defun elisa--gitignore-to-elisp-regexp (pattern)
  "Convert a .gitignore PATTERN to an Emacs Lisp regexp."
  (let ((result "")
        (i 0)
        (len (length pattern)))
    (while (< i len)
      (let ((char (aref pattern i)))
        (cond
         ;; Escape special regex characters
         ((string-match-p "[.?+*^$(){}\\[\\]\\\\]" (char-to-string char))
          (setq result (concat result "\\" (char-to-string char))))
         ;; Handle ** for any number of directories
         ((and (> len (+ i 1))
               (char-equal char ?*)
               (char-equal (aref pattern (+ i 1)) ?*))
          (setq result (concat result ".*"))
          (setq i (+ i 1)))
         ;; Handle * for any number of characters except /
         ((char-equal char ?*)
          (setq result (concat result "[^/]*")))
         ;; Handle ? for a single character except /
         ((char-equal char ??)
          (setq result (concat result "[^/]")))
         ;; Handle negation
         ((char-equal char ?!)
          (setq result (concat result "^")))
         ;; Handle directory separator
         ((char-equal char ?/)
          (setq result (concat result "/")))
         ;; Default case: add the character as is
         (t
          (setq result (concat result (char-to-string char))))))
      (setq i (+ i 1)))
    ;; prevent false-positive partial matches
    (concat result "$")))

(defun elisa--read-ignore-file-regexps (directory)
  "Read ignore patterns from `elisa-ignore-patterns-files' in DIRECTORY."
  (mapcar #'elisa--gitignore-to-elisp-regexp
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
  (or (when (get-file-buffer filename) t) ;; if file opened assume it text
      (with-current-buffer (find-file-noselect filename t t)
	(prog1
	    ;; if there is null byte in file, file is binary
	    (not (re-search-forward "\0" nil t 1))
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
		       (elisa--text-file-p file)))
		(directory-files-recursively directory ".*"))))

(defun elisa-parse-file (collection-id path &optional force)
  "Parse file PATH for COLLECTION-ID.
When FORCE parse even if already parsed."
  (let* ((opened (get-file-buffer path))
	 (buf (or opened (find-file-noselect path t t)))
	 (hash (secure-hash 'sha256 buf))
	 (prev-hash (caar (sqlite-select
			   elisa-db
			   (format "select hash from files where path = '%s';"
				   (elisa-sqlite-escape path))))))
    (when (or force
	      (not prev-hash)
	      (not (string-equal hash prev-hash)))
      (with-current-buffer buf
	(let ((chunks (elisa-split-semantically))
	      (old-row-ids
	       (flatten-tree (sqlite-select
			      elisa-db
			      (format "select rowid from data where path = '%s';"
				      (elisa-sqlite-escape path)))))
	      (row-ids nil)
	      (kind-id (caar (sqlite-select
			      elisa-db
			      "select rowid from kinds where name = 'file';"))))
	  ;; remove old data
	  (when prev-hash
	    (sqlite-execute
	     elisa-db
	     (format "delete from files where path = '%s';"
		     (elisa-sqlite-escape path))))
	  ;; add new data
	  (mapc
	   (lambda (text)
	     (let* ((hash (secure-hash 'sha256 text))
		    (rowid
		     (if-let ((rowid (caar (sqlite-select
					    elisa-db
					    (format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';"
						    kind-id collection-id
						    (elisa-sqlite-escape path) hash)))))
			 (progn
			   (push rowid row-ids)
			   nil)
		       (sqlite-execute
			elisa-db
			(format
			 "insert into data(kind_id, collection_id, path, hash, data) values (%s, %s, '%s', '%s', '%s');"
			 kind-id collection-id
			 (elisa-sqlite-escape path) hash (elisa-sqlite-escape text)))
		       (caar (sqlite-select
			      elisa-db
			      (format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';"
				      kind-id collection-id
				      (elisa-sqlite-escape path) hash))))))
	       (when rowid
		 (sqlite-execute
		  elisa-db
		  (format "insert into data_embeddings(rowid, embedding) values (%s, %s);"
			  rowid (elisa-vector-to-sqlite
				 (llm-embedding elisa-embeddings-provider text))))
		 (sqlite-execute
		  elisa-db
		  (format "insert into data_fts(rowid, data) values (%s, '%s');"
			  rowid (elisa-sqlite-escape text)))
		 (push rowid row-ids))))
	   chunks)
	  ;; remove old data
	  (when row-ids
	    (let ((delete-rows (cl-remove-if (lambda (id)
					       (cl-find id row-ids))
					     old-row-ids)))
	      (elisa--delete-data delete-rows)))
	  ;; save hash to files table
	  (sqlite-execute
	   elisa-db
	   (format "insert into files (path, hash) values ('%s', '%s');"
		   (elisa-sqlite-escape path) hash)))))
    ;; kill buffer if it was not open before parsing
    (when (not opened)
      (kill-buffer buf))))

(defun elisa--delete-data (ids)
  "Delete data with IDS."
  (sqlite-execute
   elisa-db
   (format "delete from data_fts where rowid in %s;"
	   (elisa-sqlite-format-int-list ids)))
  (sqlite-execute
   elisa-db
   (format "delete from data_embeddings where rowid in %s;"
	   (elisa-sqlite-format-int-list ids)))
  (sqlite-execute
   elisa-db
   (format "delete from data where rowid in %s;"
	   (elisa-sqlite-format-int-list ids))))

(defun elisa-parse-directory (dir)
  "Parse DIR as new collection syncronously."
  (setq dir (expand-file-name dir))
  (let* ((collection-id (progn
			  (sqlite-execute
			   elisa-db
			   (format
			    "insert into collections (name) values ('%s') on conflict do nothing;"
			    (elisa-sqlite-escape dir)))
			  (caar (sqlite-select
				 elisa-db
				 (format
				  "select rowid from collections where name = '%s';"
				  (elisa-sqlite-escape dir))))))
	 (files (elisa--file-list dir))
	 (delete-ids (flatten-tree
		      (sqlite-select
		       elisa-db
		       (format
			"select rowid from data where collection_id = %d and path not in %s;"
			collection-id
			(elisa-sqlite-format-string-list files))))))
    (elisa--delete-data delete-ids)
    (mapc (lambda (file)
	    (message "parsing %s" file)
	    (elisa-parse-file collection-id file))
	  files)))

;;;###autoload
(defun elisa-async-parse-directory (dir)
  "Parse DIR as new collection asyncronously."
  (interactive "DSelect directory: ")
  (elisa--async-do (lambda ()
		     (elisa-parse-directory
		      (expand-file-name dir)))))

(defun elisa-search-duckduckgo (prompt)
  "Search duckduckgo for PROMPT and return list of urls."
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
	:test 'string-equal)))))

(defun elisa-search-searxng (prompt)
  "Search searxng for PROMPT and return list of urls.
You can customize `elisa-searxng-url' to use non local instance."
  (let ((url (format "%s/search?format=json&q=%s" elisa-searxng-url (url-hexify-string prompt))))
    (thread-last
      (plz 'get url :as 'json-read)
      (alist-get 'results)
      (mapcar (lambda (el) (alist-get 'url el))))))

(defun elisa-get-webpage-buffer (url)
  "Get buffer with URL content."
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
       (format "%s -f html --to plain"
	       (executable-find elisa-pandoc-executable))
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
	    "select rowid, data from data where rowid in %s;"
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
			elisa-db "select rowid from kinds where name = 'web';"))))
    (message "collecting data from %s" url)
    (mapc
     (lambda (chunk)
       (let* ((hash (secure-hash 'sha256 chunk))
	      (embedding (llm-embedding elisa-embeddings-provider chunk))
	      (rowid
	       (if-let ((rowid (caar (sqlite-select
				      elisa-db
				      (format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';" kind-id collection-id url hash)))))
		   nil
		 (sqlite-execute
		  elisa-db
		  (format
		   "insert into data(kind_id, collection_id, path, hash, data) values (%s, %s, '%s', '%s', '%s');"
		   kind-id collection-id url hash (elisa-sqlite-escape chunk)))
		 (caar (sqlite-select
			elisa-db
			(format "select rowid from data where kind_id = %s and collection_id = %s and path = '%s' and hash = '%s';" kind-id collection-id url hash))))))
	 (when rowid
	   (sqlite-execute
	    elisa-db
	    (format "insert into data_embeddings(rowid, embedding) values (%s, %s);"
		    rowid (elisa-vector-to-sqlite embedding)))
	   (sqlite-execute
	    elisa-db
	    (format "insert into data_fts(rowid, data) values (%s, '%s');"
		    rowid (elisa-sqlite-escape chunk))))))
     (elisa-extact-webpage-chunks url))))

(defun elisa--web-search (prompt)
  "Search the web for PROMPT.
Return sqlite query that extract data for adding to context."
  (sqlite-execute
   elisa-db
   (format
    "insert into collections (name) values ('%s') on conflict do nothing;"
    (elisa-sqlite-escape prompt)))
  (let* ((collection-id (caar (sqlite-select
			       elisa-db
			       (format
				"select rowid from collections where name = '%s';"
				(elisa-sqlite-escape prompt)))))
	 (urls (funcall elisa-web-search-function prompt))
	 (collected-pages 0))
    (mapc (lambda (url)
	    (when (<= collected-pages elisa-web-pages-limit)
	      (elisa--parse-web-page collection-id url)
	      (cl-incf collected-pages)))
	  urls)))

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
    (lambda (s) (or (string-suffix-p ".info" s)
		    (string-suffix-p ".info.gz" s)))
    (directory-files (with-temp-buffer
		       (info "emacs" (current-buffer))
		       (file-name-directory Info-current-file))))))

(defun elisa-get-external-manuals ()
  "Get external manual names list."
  (cl-remove-if
   #'not
   (mapcar
    #'elisa--info-valid-p
    (seq-uniq
     (mapcar
      #'file-name-base
      (process-lines
       (executable-find elisa-find-executable)
       (file-truename
	(file-name-concat user-emacs-directory "elpa"))
       "-name" "*.info"))))))

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
  (let ((command real-this-command))
    (async-start `(lambda ()
		    ,(async-inject-variables "elisa-embeddings-provider")
		    ,(async-inject-variables "elisa-db-directory")
		    ,(async-inject-variables "elisa-find-executable")
		    ,(async-inject-variables "elisa-tar-executable")
		    ,(async-inject-variables "elisa-prompt-rewriting-enabled")
		    ,(async-inject-variables "elisa-rewrite-prompt-template")
		    ,(async-inject-variables "elisa-semantic-split-function")
		    ,(async-inject-variables "elisa-webpage-extraction-function")
		    ,(async-inject-variables "elisa-web-search-function")
		    ,(async-inject-variables "elisa-searxng-url")
		    ,(async-inject-variables "elisa-web-pages-limit")
		    ,(async-inject-variables "elisa-breakpoint-threshold-amount")
		    ,(async-inject-variables "elisa-pandoc-executable")
		    ,(async-inject-variables "ellama-long-lines-length")
		    ,(async-inject-variables "elisa-reranker-enabled")
		    ,(async-inject-variables "load-path")
		    ,(async-inject-variables "Info-directory-list")
		    ,(async-inject-variables "elisa-sqlite-vector-path")
		    ,(async-inject-variables "elisa-sqlite-vss-path")
		    (require 'elisa)
		    (,func))
		 (lambda (res)
		   (sqlite-close elisa-db)
		   (elisa--reopen-db)
		   (when on-done
		     (funcall on-done res))
		   (message "%s done."
			    (or command "async elisa processing"))))))

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
			     "select name from collections;")))
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
(defun elisa-disble-all-collections ()
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
		     "select name from collections;")))))))
    (push col elisa-enabled-collections)))

;;;###autoload
(defun elisa-create-empty-collection (&optional collection)
  "Create new empty COLLECTION."
  (interactive "sNew collection name: ")
  (save-window-excursion
    (sqlite-execute
     elisa-db
     (format
      "insert into collections (name) values ('%s') on conflict do nothing;"
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
       "select name from collections;")))))
  (let ((collection-id (caar (sqlite-select
			      elisa-db
			      (format
			       "select rowid from collections where name = '%s';"
			       (elisa-sqlite-escape collection))))))
    (elisa--async-do (lambda () (elisa-parse-file collection-id file)))))

;;;###autoload
(defun elisa-add-webpage-to-collection (url collection)
  "Add webpage by URL to COLLECTION."
  (interactive
   (list
    (if-let ((url (or (and (fboundp 'thing-at-point) (thing-at-point 'url))
                      (shr-url-at-point nil))))
        url
      (read-string "Enter URL you want to summarize: "))
    (completing-read
     "Enable collection: "
     (flatten-tree
      (sqlite-select
       elisa-db
       "select name from collections;")))))
  (let ((collection-id (caar (sqlite-select
			      elisa-db
			      (format
			       "select rowid from collections where name = '%s';"
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
		     "select name from collections;")))))
	 (collection-id (caar (sqlite-select
			       elisa-db
			       (format
				"select rowid from collections where name = '%s';"
				(elisa-sqlite-escape col)))))
	 (delete-ids (flatten-tree
		      (sqlite-select
		       elisa-db
		       (format
			"select rowid from data where collection_id = %d;"
			collection-id)))))
    (elisa-disable-collection col)
    (when (file-directory-p col)
      (let ((files
	     (flatten-tree
	      (sqlite-select
	       elisa-db
	       (format
		"select distinct path from data where collection_id = %d;"
		collection-id)))))
	(sqlite-execute
	 elisa-db
	 (format
	  "delete from files where path in %s;"
	  (elisa-sqlite-format-string-list files)))))
    (elisa--delete-data delete-ids)
    (sqlite-execute
     elisa-db
     (format
      "delete from collections where rowid = %d;"
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

(provide 'elisa)
;;; elisa.el ends here.
