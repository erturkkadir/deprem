<?php
  header('Content-Type: application/json');
  header('Access-Control-Allow-Origin: *');
  header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
  header('Access-Control-Allow-Headers: Content-Type');

  if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
      exit;
  }

  $path = $_SERVER['PATH_INFO'] ?? '';
  $query = $_SERVER['QUERY_STRING'] ?? '';
  $api_url = 'http://vanc.syshuman.com:3000/api' . $path;
  if ($query) {
      $api_url .= '?' . $query;
  }

  if ($_SERVER['REQUEST_METHOD'] === 'POST') {
      $data = file_get_contents('php://input');
      $opts = ['http' => ['method' => 'POST', 'header' => 'Content-Type: application/json', 'content' => $data]];
      echo file_get_contents($api_url, false, stream_context_create($opts));
  } else {
      echo file_get_contents($api_url);
  }