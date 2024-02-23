;;; elisa.el --- Emacs Lisp Information System Assistant -*- lexical-binding: t -*-

;; Copyright (C) 2024  Free Software Foundation, Inc.

;; Author: Sergey Kostyaev <sskostyaev@gmail.com>
;; URL: http://github.com/s-kostyaev/elisa
;; Keywords: help local tools
;; Package-Requires: ((emacs "29.2") (ellama "0.8.5") (llm "0.9.1") (async "1.9.8"))
;; Version: 0.1.0
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
;; ELISA (Emacs Lisp Information System Assistant) is a project
;; designed to help Emacs users quickly find answers to their
;; questions related to Emacs and Emacs Lisp. Utilizing the powerful
;; Ellama package, ELISA provides accurate and relevant responses to
;; user queries, enhancing productivity and efficiency in the Emacs
;; environment. By integrating links to the Emacs info manual after
;; answering a question, ELISA ensures that users have easy access to
;; additional information on the topic, making it an essential tool
;; for both beginners and advanced Emacs users.
;;

;;; Code:
(require 'ellama)
(require 'llm)
(require 'info)
(require 'async)

(defcustom elisa-embeddings-provider nil
  "Embeddings provider to generate embeddings."
  :group 'tools
  :type '(sexp :validate 'cl-struct-p))

(defcustom elisa-db-directory (file-truename
			       (file-name-concat
				user-emacs-directory "elisa"))
  "Directory for elisa database."
  :group 'tools
  :type 'directory)

(defcustom elisa-limit 7
  "Count info nodes to pass into llm context for answer."
  :group 'tools
  :type 'integer)

(defcustom elisa-find-executable (executable-find "find")
  "Path to find executable."
  :group 'tools
  :type 'integer)

(defcustom elisa-tar-executable (executable-find "tar")
  "Path to tar executable."
  :group 'tools
  :type 'integer)

(defcustom elisa-sqlite-vss-version "v0.1.2"
  "Sqlite VSS version."
  :group 'tools
  :type 'string)

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
  (let* ((ext (if (string-equal system-type "darwin")
		  "dylib"
		"so"))
	 (file (format "vss0.%s" ext)))
    (file-name-concat elisa-db-directory file)))

(defun elisa--vector-path ()
  "Path to vector sqlite extension."
  (let* ((ext (if (string-equal system-type "darwin")
		  "dylib"
		"so"))
	 (file (format "vector0.%s" ext)))
    (file-name-concat elisa-db-directory file)))

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
    (process-lines elisa-tar-executable "-xf" file-name)
    (delete-file file-name)))

(defun elisa-get-embedding-size ()
  "Get embedding size."
  (length (llm-embedding elisa-embeddings-provider "test")))

(defun elisa-embeddings-create-table-sql ()
  "Generate sql for create embeddings table."
  (format "create virtual table if not exists elisa_embeddings using vss0(embedding(%d));"
	  (elisa-get-embedding-size)))

(defun elisa-info-create-table-sql ()
  "Generate sql for create info table."
  "create table if not exists info (node text unique);")

(defun elisa--init-db (db)
  "Initialize elisa DB."
  (when (not (file-exists-p (elisa--vss-path)))
    (elisa-download-sqlite-vss))
  (sqlite-load-extension
   db
   (elisa--vector-path))
  (sqlite-load-extension
   db
   (elisa--vss-path))
  (sqlite-execute db (elisa-embeddings-create-table-sql))
  (sqlite-execute db (elisa-info-create-table-sql)))

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
  (string-replace "'" "''" s))

(defun elisa-parse-info-manual (name)
  "Parse info manual with NAME and save index to database."
  (with-temp-buffer
    (info name (current-buffer))
    (let ((continue t))
      (while continue
	(let* ((node-name (concat "(" (file-name-sans-extension
				       (file-name-nondirectory Info-current-file))
				  ") "
				  Info-current-node))
	       (content (buffer-substring-no-properties (point-min) (point-max)))
	       (embedding (llm-embedding elisa-embeddings-provider content))
	       (rowid (progn
			(sqlite-execute elisa-db
					(format
					 "insert into info values('%s') on conflict do nothing;"
					 (elisa-sqlite-escape node-name)))
			(caar
			 (sqlite-select
			  elisa-db
			  (format "select rowid from info where node='%s';"
				  (elisa-sqlite-escape node-name)))))))
	  (when (not (caar
		      (sqlite-select
		       elisa-db
		       (format "select rowid from elisa_embeddings where rowid=%s;" rowid))))
	    (sqlite-execute
	     elisa-db
	     (format "insert into elisa_embeddings(rowid, embedding) values (%s, %s);"
		     rowid
		     (elisa-vector-to-sqlite embedding))))
	  (condition-case nil
	      (progn (funcall-interactively #'Info-forward-node)
		     (sleep-for 0 100))
	    (error
	     (setq continue nil))))))))

(defun elisa-find-similar (text)
  "Find similar to TEXT results."
  (let ((embedding (llm-embedding elisa-embeddings-provider text)))
    (flatten-tree
     (sqlite-select
      elisa-db
      (format
       "select * from info where rowid in
(select rowid from elisa_embeddings where vss_search(embedding,%s) limit %d);"
       (elisa-vector-to-sqlite embedding)
       elisa-limit)))))

(defun elisa-get-builtin-manuals ()
  "Get builtin manual names list."
  (mapcar
   #'file-name-base
   (cl-remove-if-not
    (lambda (s) (string-suffix-p ".info" s))
    (directory-files (with-temp-buffer
		       (info "emacs" (current-buffer))
		       default-directory)))))

(defun elisa-get-external-manuals ()
  "Get external manual names list."
  (seq-uniq
   (mapcar
    #'file-name-base
    (process-lines
     elisa-find-executable
     (file-truename
      (file-name-concat user-emacs-directory "elpa")) "-name" "*.info"))))

(defun elisa-parse-builtin-manuals ()
  "Parse builtin manuals."
  (mapc (lambda (s)
	  (ignore-errors (elisa-parse-info-manual s)))
	(elisa-get-builtin-manuals)))

(defun elisa-parse-external-manuals ()
  "Parse external manuals."
  (mapc (lambda (s)
	  (ignore-errors (elisa-parse-info-manual s)))
	(elisa-get-external-manuals)))

(defun elisa-parse-all-manuals ()
  "Parse all manuals."
  (elisa-parse-builtin-manuals)
  (elisa-parse-external-manuals))

;;;###autoload
(defun elisa-async-parse-builtin-manuals ()
  "Parse builtin manuals asyncronously."
  (interactive)
  (message "Begin parsing builtin manuals.")
  (async-start `(lambda ()
		  ,(async-inject-variables "elisa-embeddings-provider")
		  (package-initialize)
		  (require 'elisa)
		  (elisa-parse-builtin-manuals))
	       (lambda (_)
		 (message "Builtin manuals parsing done."))))

;;;###autoload
(defun elisa-async-parse-external-manuals ()
  "Parse external manuals asyncronously."
  (interactive)
  (message "Begin parsing external manuals.")
  (async-start `(lambda ()
		  ,(async-inject-variables "elisa-embeddings-provider")
		  (package-initialize)
		  (require 'elisa)
		  (elisa-parse-external-manuals))
	       (lambda (_)
		 (message "External manuals parsing done."))))

;;;###autoload
(defun elisa-async-parse-all-manuals ()
  "Parse all manuals asyncronously."
  (interactive)
  (message "Begin parsing manuals.")
  (async-start `(lambda ()
		  ,(async-inject-variables "elisa-embeddings-provider")
		  (package-initialize)
		  (require 'elisa)
		  (elisa-parse-all-manuals))
	       (lambda (_)
		 (message "Manuals parsing done."))))

;;;###autoload
(defun elisa-chat (prompt)
  "Send PROMPT to elisa."
  (interactive "sAsk elisa: ")
  (let ((infos (elisa-find-similar prompt)))
    (mapc #'ellama-context-add-info-node infos)
    (ellama-chat prompt)))

(provide 'elisa)
;;; elisa.el ends here.
