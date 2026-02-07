<?php
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

  $opts = ['http' => ['method' => $_SERVER['REQUEST_METHOD'], 'ignore_errors' => true]];
  if ($_SERVER['REQUEST_METHOD'] === 'POST') {
      $data = file_get_contents('php://input');
      $opts['http']['header'] = 'Content-Type: application/json';
      $opts['http']['content'] = $data;
  }

  $response = file_get_contents($api_url, false, stream_context_create($opts));

  // Detect content type from backend response headers
  $content_type = 'application/json';
  if (isset($http_response_header)) {
      foreach ($http_response_header as $h) {
          if (stripos($h, 'Content-Type:') === 0) {
              $content_type = trim(substr($h, 13));
              break;
          }
      }
  }
  header('Content-Type: ' . $content_type);

  echo $response;
